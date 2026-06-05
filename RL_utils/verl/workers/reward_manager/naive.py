# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Copyright 2024 Bytedance Ltd. and/or its affiliates


from collections import defaultdict
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("naive")
class NaiveRewardManager(AbstractRewardManager):
    """The reward manager with concurrent processing support."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer  
        self.num_examine = num_examine  
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}
        batch_size = len(data)
        
        # --- Step 1: Preprocess all data and prepare task dictionaries ---
        tasks = []
        for i in range(batch_size):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
            extra_info["num_turns"] = num_turns
            extra_info["rollout_reward_scores"] = rollout_reward_scores
            
            # Pack the required information into the tasks list
            tasks.append({
                "index": i,
                "prompt_str": prompt_str,
                "response_str": response_str,
                "ground_truth": ground_truth,
                "data_source": data_source,
                "extra_info": extra_info,
                "valid_response_length": valid_response_length
            })

        # --- Step 2: Define single task execution function ---
        def _process_single_task(task):
            score = self.compute_score(
                data_source=task["data_source"],
                solution_str=task["response_str"],
                ground_truth=task["ground_truth"],
                extra_info=task["extra_info"],
            )
            return task["index"], score

        # --- Step 3: Multi-thread concurrent requests ---
        scores_result = [None] * batch_size
        # max_workers can be adjusted based on your concurrency tolerance, 32 to 64 is generally suitable for LLM API requests
        with ThreadPoolExecutor(max_workers=64) as executor:
            # Submit all tasks to the thread pool
            futures = [executor.submit(_process_single_task, task) for task in tasks]
            
            # Wait for and collect results
            for future in as_completed(futures):
                idx, score = future.result()  # If the request throws an exception, it will raise an error and terminate here
                scores_result[idx] = score

        # --- Step 4: Merge results, fill Tensor, and print logs ---
        for task in tasks:
            i = task["index"]
            score = scores_result[i]
            valid_response_length = task["valid_response_length"]
            data_source = task["data_source"]
            prompt_str = task["prompt_str"]
            response_str = task["response_str"]
            ground_truth = task["ground_truth"]

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            # Log printing logic remains unchanged
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor