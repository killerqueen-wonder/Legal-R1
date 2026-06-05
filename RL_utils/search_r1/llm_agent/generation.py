import torch
import re
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.utils.tracking import Tracking
import shutil
import requests
import json
import requests
from concurrent.futures import ThreadPoolExecutor

@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool=False
    search_url: str = None
    topk: int = 3

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to stop at search operation or answer operation."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        responses_str = [resp.split('</search>')[0] + '</search>'
                 if '</search>' in resp 
                 else resp.split('</answer>')[0] + '</answer>'
                 if '</answer>' in resp 
                 else resp
                 for resp in responses_str]

        if self.config.no_think_rl:
            raise ValueError('stop')
            # if no_think_rl is enabled, only keep action in the str
            actions, _ = self.env.postprocess_predictions(responses_str)
            responses_str=[f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>" for idx, action in enumerate(actions)]
            print("RESPONSES:", responses_str)
        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    # def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
    #     """Process next observations from environment."""
        
    #     next_obs_ids = self.tokenizer(
    #         next_obs, 
    #         padding='longest',
    #         return_tensors='pt',
    #         add_special_tokens=False,  # Prevents adding special tokens
    #     )['input_ids']

    #     if next_obs_ids.shape[1] > self.config.max_obs_length:
    #         print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")            
    #         next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

    #     return next_obs_ids
    

    def _process_next_obs(self, next_obs: List[str], device: torch.device) -> torch.Tensor:
        """
        Process environment observations:
        1. Dynamic truncation: If too long, truncate and append </information>.
        2. Dynamic padding: Only pad to the maximum length in the current batch, significantly saving VRAM.
        3. Device alignment: Directly move the result to the specified training device.
        """
        # Preset label information
        suffix = "</information>"
        suffix_token_ids = self.tokenizer(suffix, add_special_tokens=False)['input_ids']
        suffix_len = len(suffix_token_ids)
        
        # Get the configured hard limit
        max_limit = self.config.max_obs_length
        max_content_len = max_limit - suffix_len

        processed_tensors = []
        
        for obs_text in next_obs:
            # Single tokenize, do not pad here
            full_ids = self.tokenizer(obs_text, add_special_tokens=False)['input_ids']

            # Execute truncation and completion logic
            if len(full_ids) > max_limit:
                final_ids = full_ids[:max_content_len] + suffix_token_ids
            else:
                final_ids = full_ids
            
            # Create CPU Tensor (default), very low memory usage at this point
            processed_tensors.append(torch.tensor(final_ids, dtype=torch.long))

        # Execute dynamic padding (only pad to the longest sequence in this batch)
        from torch.nn.utils.rnn import pad_sequence
        next_obs_ids = pad_sequence(
            processed_tensors, 
            batch_first=True, 
            padding_value=self.tokenizer.pad_token_id
        )

        # Move from CPU to the specified "borrowed" device (GPU)
        return next_obs_ids.to(device)

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding        
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)
        
        return new_rollings

    def _info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor, 
                prompt_with_mask: torch.Tensor, 
                response: torch.Tensor, 
                info: torch.Tensor = None,
                pad_to_left: bool = True
            ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            tensors_with_mask.append(info_mask)
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        if next_obs_ids != None:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    next_obs_ids, 
                    pad_to_left=False
                )
        else:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    pad_to_left=False
                )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""
        
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []]}
        
        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        turns_stats = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_search_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch

        
        # Main generation loop
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })      
            # Dictionary comprehension, iterate over each key-value pair (k, v) in rollings.batch, and use active_mask to index value v, keeping only the data of currently active samples. This effectively creates a sub-batch containing only active samples.
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # Execute in environment and process observations
            next_obs, dones, valid_action, is_search = self.execute_predictions(
                predictions=responses_str, 
                pad_token=self.tokenizer.pad_token, 
                active_mask=active_mask, 
                do_search=True,
                current_turn=step
            )
            
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)

            # Get the device where the current batch is located, return after next_obs processing
            current_device = rollings.batch['input_ids'].device

            next_obs_ids = self._process_next_obs(next_obs,current_device)
            
            # Update states
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids
            )
            
        # final LLM rollout
        if active_mask.sum():
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })            
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # # Execute in environment and process observations
            _, dones, valid_action, is_search = self.execute_predictions(
                predictions=responses_str, 
                pad_token=self.tokenizer.pad_token, 
                active_mask=active_mask, 
                do_search=False,
                current_turn=self.config.max_turns
            )

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)
            

            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
            )
        
        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        meta_info['valid_search_stats'] = valid_search_stats.tolist()
        
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        
        return self._compose_final_output(original_left_side, original_right_side, meta_info)

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        
        return final_output

    def execute_predictions(self, predictions: List[str], pad_token: str, active_mask=None, do_search=True,current_turn=-1) -> List[str]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
        
        Args:
            envs: List of environment instances
            predictions: List of action predictions
            pad_token: Token to use for padding
            
        Returns:
            List of observation strings
        """
        cur_actions, contents, contexts = self.postprocess_predictions(predictions)
        next_obs, dones, valid_action, is_search = [], [], [], []
        current_turn+=1
        # Calculate maximum turns
        max_turns = self.config.max_turns
        # Construct turn reminder text
        turn_info = f"\n[系统提示] 当前为第 {current_turn} 轮检索（上限 {max_turns} 轮）。"
        if current_turn >= max_turns:
            # print('[debug] Reached search limit.')
            turn_info += " 以下是最后一次检索结果。注意：接下来总结以上思考，必须给出最终回答！"

        # search_queries = [content for action, content in zip(cur_actions, contents) if action == 'search']
        # Extract search queries and corresponding contexts simultaneously
        search_pairs = [(cont, ctx) for action, cont, ctx in zip(cur_actions, contents, contexts) if action == 'search']
        search_queries = [p[0] for p in search_pairs]
        # Extract the corresponding context list
        search_contexts = [p[1] for p in search_pairs] 
        if do_search:
            search_results = self.batch_search(search_queries, search_contexts)
            assert len(search_results) == sum([1 for action in cur_actions if action == 'search'])
        else:
            search_results = ['接下来总结以上思考，必须给出最终回答！'] * sum([1 for action in cur_actions if action == 'search'])

        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):

            if not active:
                next_obs.append('')
                dones.append(1)
                valid_action.append(0)
                is_search.append(0)
            else:
                
                if action == 'answer':
                    next_obs.append('')
                    dones.append(1)
                    valid_action.append(1)
                    is_search.append(0)
                elif action == 'search':

                    next_obs.append(f'\n{turn_info}\n<information>{search_results.pop(0).strip()}</information>\n\n')
                    dones.append(0)
                    valid_action.append(1)
                    is_search.append(1)
                else:
                    next_obs.append(f'\n【系统提示】 \
如果下一步需要搜索，应该把搜索内容放在<search> 和 </search>之间。 \
如果下一步给出最终回答，应该把答案放在 <answer> 和 </answer>之间。让我重新思考。【/系统提示】\n')
                    dones.append(0)
                    valid_action.append(0)
                    is_search.append(0)
            
        assert len(search_results) == 0
            
        return next_obs, dones, valid_action, is_search


    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[int], List[bool]]:
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, validity flags list)
        """
        actions = []
        contents = []
        contexts = [] # Added: Store content outside the tags
                
        for prediction in predictions:
            if isinstance(prediction, str): # for llm output
                pattern = r'<(search|answer)>(.*?)</\1>'
                match = re.search(pattern, prediction, re.DOTALL)
                if match:
                    content = match.group(2).strip()  # Return only the content inside the tags
                    action = match.group(1)
                    context = prediction[:match.start()].strip()
                else:
                    content = ''
                    action = None
                    context = ''
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            contents.append(content)
            contexts.append(context)
            
        # return actions, contents
        return actions, contents, contexts



    def local_rag_search(self, search_json_str: str) -> str:
        """
        Core RAG parsing and request method:
        Parses the <search> JSON generated by the LLM, sends it to the RAG service, and formats the returned result according to the synthetic data script logic.
        """
        try:
            
            
            clean_str = search_json_str.strip()
            # 1. Strip possible Markdown code blocks (e.g., ```json and ```)
            if clean_str.startswith("```json"):
                clean_str = clean_str[7:]
            elif clean_str.startswith("```"):
                clean_str = clean_str[3:]
            if clean_str.endswith("```"):
                clean_str = clean_str[:-3]
                
            clean_str = clean_str.strip()
            
            # 2. Fault-tolerant replacement of Chinese punctuation
            clean_str = clean_str.replace("，", ",")        # Chinese comma to English comma
            clean_str = clean_str.replace("“", '"')       # Chinese left double quote to English double quote
            clean_str = clean_str.replace("”", '"')       # Chinese right double quote to English double quote
            
            # 3. Structure repair: auto-complete missing curly braces at the beginning and end
            if not clean_str.startswith("{"):
                clean_str = "{" + clean_str
            if not clean_str.endswith("}"):
                clean_str = clean_str + "}"
            # ================= End of JSON robustness cleaning =================
            search_query = json.loads(clean_str)
            

            # 2. Assemble UnifiedQueryRequest payload for FastAPI
            payload = {
                "query": search_query,
                "topk": self.config.topk
            }
            
            # 3. Send request (set 120s long timeout because RAG queues during high concurrency)
            response = requests.post(
                self.config.search_url,
                json=payload,
                proxies={"http": None, "https": None},
                timeout=120.0
            )
            response.raise_for_status()
            json_data = response.json()
            
            # 4. Fault tolerance processing
            if "error" in json_data:
                return f"检索返回错误：{json_data['error']}"

            # 5. Disassemble response according to synthetic data format
            req_type = json_data.get("检索类型", "")

            if req_type == "类案检索":
                summary = json_data.get("llm_summary", "未检索到匹配的类案分析结果。")
                return f"【类案检索分析报告】\n{summary}"

            elif req_type == "法律检索":
                results = json_data.get("result", [])
                format_reference = []
                
                for idx, doc_item in enumerate(results):
                    # Get content and score
                    content = doc_item.get('document', {}).get('content', '')
                    score = doc_item.get('score', 0.0)
                    # Numbering starts from 1, corresponding to [法条参考 X] in the prompt
                    format_reference.append(f"法条参考 {idx + 1} (相关度: {score:.4f}):\n{content}\n")
                
                if not format_reference:
                    return "【法律检索结果】未找到相关的法律条文，请尝试更换关键词。"
                return "【法律检索结果】\n" + "\n".join(format_reference)
            else:
                return f"未知的检索类型返回，原始数据: {str(json_data)[:200]}"
                
        except json.JSONDecodeError as e:
            return "工具调用失败：<search>标签内的JSON格式不合法，请检查并输出合法的JSON格式再试一次。"
        except Exception as e:
            return f"检索请求出错: {str(e)}"

    def batch_search(self, queries: List[str] = None, contexts: List[str] = None) -> List[str]:
        """
        Use a thread pool to concurrently execute batch-level search requests, perfectly matching the environment interaction loop.
        """
        if not queries:
            return []
            
        # Concurrency is the current number of search requests
        with ThreadPoolExecutor(max_workers=len(queries)) as executor:
            # map keeps the input list and output list in the same order, which is crucial for aligning reinforcement learning trajectories
            results = list(executor.map(self.local_rag_search, queries))
            
        return results