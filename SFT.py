import os
import json
import torch
import logging
import argparse
import re
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import wandb
from accelerate import Accelerator
from transformers import set_seed, get_cosine_schedule_with_warmup
import shutil
import json
import traceback
from jinja2 import Template

from transformers import AutoModelForCausalLM, AutoTokenizer
os.umask(0)


logger = logging.getLogger(__name__)
logging.basicConfig(level='INFO')




class Train_dataset(torch.utils.data.Dataset):
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        with open(config.data_path) as f:
            raw_data = json.load(f)

        self.data = []
        required_keys = ['Open-ended Verifiable Question', 'Complex_CoT']
        for da in raw_data:
            if all(k in da and da[k] is not None for k in required_keys):
                self.data.append(da)
        
        print(f'[Dataset] original data size: {len(raw_data)}, filter data size: {len(self.data)}')

        self.max_seq_len = self.config.max_seq_len
        self.debug = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def get_response(self, da):
        # temp = '## Thinking\n\n{}\n\n## Final Response\n<answer>\n{}\n</answer>'
        # return temp.format(da['Complex_CoT'], da['Response'])
        return (da['Complex_CoT'])

    def get_prompt(self, da):
        
        try:
            q = da['Open-ended Verifiable Question']
            a = self.get_response(da)

            messages = [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a}
            ]
            
            full_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            query_prompt = self.tokenizer.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
            assistant_start_idx = len(query_prompt)

            encoding = self.tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
            input_ids = encoding['input_ids']
            offsets = encoding['offset_mapping']
            
            mask_spans = []
            for match in re.finditer(r'<information>.*?</information>', full_text, re.DOTALL):
                mask_spans.append(match.span())

            labels = []
            for i, (start, end) in enumerate(offsets):
                if start < assistant_start_idx:
                    labels.append(-100)
                    continue
                
                is_in_info_block = False
                for s_start, s_end in mask_spans:
                    if max(start, s_start) < min(end, s_end):
                        is_in_info_block = True
                        break
                labels.append(-100 if is_in_info_block else input_ids[i])

            if len(input_ids) > self.max_seq_len:
                input_ids = input_ids[:self.max_seq_len]
                labels = labels[:self.max_seq_len]

            return {"input_ids": input_ids, "labels": labels}

        except Exception as e:
            print(f"Error processing sample: {e}")
            return {"input_ids": [self.tokenizer.eos_token_id], "labels": [-100]}

    def collate_fn(self, batch):
        processed_batch = []
        for da in batch:
            item = self.get_prompt(da)
            processed_batch.append(item)

        input_ids = [torch.tensor(item["input_ids"]) for item in processed_batch]
        labels = [torch.tensor(item["labels"]) for item in processed_batch]

        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.eos_token_id
        )
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        

        if self.debug < 1:
            # only print in Rank 0
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f'\n{"="*20} Debug Sample {self.debug} {"="*20}')
                
                sample_input = input_ids_padded[-1].tolist()
                sample_label = labels_padded[-1].tolist()

                full_text = self.tokenizer.decode(sample_input, skip_special_tokens=False)
                
                learnable_text = self.tokenizer.decode([0 if x == -100 else x for x in sample_label])

                print(f'Final Input:\n{full_text}') 
                print(f'\nWhat Model Learns (Labels):\n{learnable_text}')
                print(f'{"="*55}\n')
                
                self.debug += 1

        return {
            "input_ids": input_ids_padded,
            "labels": labels_padded,
        }

class SFTMetric:
    def __init__(self, device):
        self.n_step = 0
        self.right = torch.Tensor([0]).to(device=device)
        self.total = torch.Tensor([0]).to(device=device)
        self.total_loss = torch.Tensor([0]).to(device=device)
        self.world_size = dist.get_world_size()

    def __call__(self, logits, labels, loss):
        return self.update(logits, labels, loss)

    def update(self, logits, labels, loss):
        self.n_step += 1
        with torch.no_grad():
            shift_preds = logits[..., :-1, :].argmax(dim=-1)
            shift_labels = labels[..., 1:]
            self.right += (shift_preds == shift_labels).masked_fill(shift_labels.eq(-100), 0).sum().item()
            self.total += (shift_labels != -100).sum().item()
            self.total_loss += loss.item()

    def get_metric(self, reset=True):
        dist.all_reduce(self.right, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(self.total, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(self.total_loss, op=torch.distributed.ReduceOp.SUM)

        acc = (self.right / self.total).item()
        loss = self.total_loss.item() / (self.world_size * self.n_step)

        if reset:
            self.n_step = 0
            self.right.fill_(0)
            self.total.fill_(0)
            self.total_loss.fill_(0)
        return acc, loss


def train(args):

    accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=args.gradient_accumulation_steps) 

    if accelerator.is_main_process:
        wandb.init(project = args.experiment_name, config=args, dir=args.log_dir, mode="online")
    
    accelerator.print(f'args:\n{args}')

    accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = args.train_bsz_per_gpu
    accelerator.state.deepspeed_plugin.deepspeed_config['train_batch_size'] = args.train_bsz_per_gpu*dist.get_world_size()*accelerator.gradient_accumulation_steps

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(args.model_path,
                                                 torch_dtype="auto",
                                                #  device_map="auto", 
                                                 trust_remote_code=True)

    # open gradient checkpointing
    model.gradient_checkpointing_enable()

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    train_dataset = Train_dataset(args, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_bsz_per_gpu, shuffle=True, drop_last=True, collate_fn=train_dataset.collate_fn)

    num_training_steps = int(len(train_dataloader) * (args.n_epochs)) // accelerator.gradient_accumulation_steps
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_rates * num_training_steps), num_training_steps=num_training_steps)
    accelerator.print(f'gradient_accumulation_steps:{accelerator.gradient_accumulation_steps} data_path:{args.data_path} lr:{args.learning_rate} num_training_steps:{num_training_steps}')
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    start_epoch = 0
    start_step = 0
    global_step = 0

    metric = SFTMetric(device=torch.cuda.current_device())

    def save_checkpoint(epoch, step, global_step):

        save_dir = os.path.join(args.output_dir, f"checkpoint-{epoch}-{global_step}")
        accelerator.wait_for_everyone()
        state_dict = accelerator.get_state_dict(model)

        if accelerator.is_main_process:

            checkpoint_files = [f for f in os.listdir(args.output_dir) if f.startswith("checkpoint-")]
            if args.max_ckpts > 0 and len(checkpoint_files) >= args.max_ckpts:
                checkpoint_files.sort(key=lambda x: os.path.getctime(os.path.join(args.output_dir, x)))
                oldest_checkpoint = checkpoint_files[0]
                shutil.rmtree(os.path.join(args.output_dir, oldest_checkpoint), ignore_errors=True)

            os.makedirs(save_dir, exist_ok=True)
            output_dir = os.path.join(save_dir, "tfmr")

            # only main process saves ckp
            unwrap_model = accelerator.unwrap_model(model)
            unwrap_model.save_pretrained(
                output_dir,
                state_dict=state_dict,
                save_function=accelerator.save,
                safe_serialization=True
            )

            tokenizer.save_pretrained(output_dir)

            # copy support files
            copy_files = []
            for item in os.listdir(args.model_path):
                if os.path.exists(os.path.join(output_dir, item)):
                    continue
                if item.startswith("pytorch_model") and item.endswith(".bin"):
                    continue
                if item.endswith(".index.json") or item.endswith(".safetensors"):
                    continue
                src = os.path.join(args.model_path, item)
                if os.path.isfile(src):
                    shutil.copy(src, os.path.join(output_dir, item))
                    copy_files.append(item)

            print(f"HuggingFace model saved in {output_dir}, copied: {copy_files}")

        accelerator.wait_for_everyone()

        accelerator.save(
            {"epoch": epoch, "step": step, "global_step": global_step},
            os.path.join(save_dir, "training_state.pt")
        )

        accelerator.print(f"Checkpoint checkpoint-{epoch}-{global_step} saved.")


    accelerator.print(accelerator.deepspeed_config)
    model.train()

    for epoch in range(start_epoch, args.n_epochs):
        train_dataloader_iterator = tqdm(enumerate(train_dataloader), total=len(train_dataloader)) if accelerator.is_main_process else enumerate(train_dataloader)
        for batch_cnt, batch in train_dataloader_iterator:
            if epoch==start_epoch and batch_cnt<start_step:
                continue

            if batch_cnt == 1 and epoch == 0:
                torch.cuda.empty_cache()

            input_ids=batch['input_ids']
            labels=batch['labels']

            output = model(input_ids=input_ids, labels=labels, return_dict=True,use_cache=False)
            loss = output.loss

            metric(output.logits, labels, loss)
            acc, train_loss = metric.get_metric()
            accelerator.backward(loss)
            if (global_step+1) % accelerator.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            global_step += 1

            if accelerator.is_main_process:
                train_dataloader_iterator.set_postfix(epoch=epoch, current_step=batch_cnt, total_step=len(train_dataloader), skip=accelerator.optimizer_step_was_skipped, loss=round(train_loss, 3), acc=round(acc, 3), length=len(input_ids[0]), lr=lr_scheduler.get_last_lr()[0])

            if global_step % 3 == 0 and accelerator.is_main_process:
                wandb.log({
                    'skip': int(accelerator.optimizer_step_was_skipped),
                    'loss': train_loss,
                    'acc': acc,
                    'lr': lr_scheduler.get_last_lr()[0]
                }, step=global_step)

        accelerator.wait_for_everyone()
        save_checkpoint(epoch, batch_cnt, global_step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args of sft')
    parser.add_argument('--experiment_name', type=str,default='sft_stage1')
    parser.add_argument('--model_path', required=True, type=str)
    parser.add_argument('--data_path', required=True, type=str)

    # Training Args
    parser.add_argument('--output_dir', default='./ckpts', type=str)
    parser.add_argument('--max_ckpts', default=2, type=int)
    parser.add_argument('--log_dir', default='./train_logs', type=str)
    parser.add_argument('--max_seq_len', default=8192, type=int)
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument('--gradient_accumulation_steps', default=8, type=int)
    parser.add_argument('--train_bsz_per_gpu', default=1, type=int)
    parser.add_argument('--weight_decay', default=0.1, type=float)
    parser.add_argument('--learning_rate', default=5e-6, type=float)
    parser.add_argument('--warmup_rates', default=0.05, type=float)
    parser.add_argument('--n_epochs', default=3, type=int)

    # Other Args
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()
    args.log_dir = os.path.join(args.log_dir,args.experiment_name)
    args.output_dir = os.path.join(args.output_dir,args.experiment_name)

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)
    train(args)           
