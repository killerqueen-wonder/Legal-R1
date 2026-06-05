# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 Search-R1 Contributors
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
# Adapted from https://github.com/PeterGriffinJin/Search-R1/blob/main/verl/utils/reward_score/qa_em.py
import random
import re
import string
import unicodedata

def normalize_answer(s):
    if s is None: # Defensive programming: handle None input
        return ""
    
    def white_space_fix(text):
        return " ".join(text.split())

    
    def remove_punc_keep_numeric(text):
        numeric_whitelist = {'.', '-', '%'}
        res = []
        for char in text:
            cat = unicodedata.category(char)
            if char.isalnum() or char.isspace():
                res.append(char)
            elif cat.startswith('P'):
                if char in numeric_whitelist:
                    res.append(char)
                else:
                    res.append(" ") # Replace other punctuation with spaces to avoid concatenation
        return "".join(res)

    def lower(text):
        return text.lower()

    # Removed remove_articles which causes ABCD errors
    return white_space_fix(remove_punc_keep_numeric(lower(s)))

def em_check(prediction, golden_answers, max_score=1.0):
    if prediction is None:
        return 0
    
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    
    # Normalize ground truth answers
    gold_set = {normalize_answer(g) for g in golden_answers if normalize_answer(g)}

    # Normalize predicted answer
    norm_pred = normalize_answer(prediction)
    
    # Strategy: If none of the items in the standard answer contain spaces, split by space for set comparison (suitable for A B C D)
    # Otherwise, try comma splitting or full match (suitable for phrases)
    if all(" " not in g for g in gold_set):
        pred_set = set(norm_pred.split())
    else:
        # If it contains phrases, it is recommended that the model output is comma-separated. Here we do simple compatibility handling.
        pred_set = {p.strip() for p in re.split(r'[,，]', norm_pred) if p.strip()}

    if not pred_set:
        return 0

    if pred_set == gold_set:
        return max_score
    
    if pred_set.issubset(gold_set):
        return max_score / 5
    
    return 0

def extract_solution(solution_str):
    if not solution_str:
        return None
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    if not matches:
        return None
    return matches[-1].group(1).strip()



def correct_format(text):
    
    return all([
        text.count("<answer>") <= 4, 
        text.count("</answer>") <= 4,
        text.count("<search>") == text.count("</search>"),
        text.count("<search>") == text.count("<information>"),
        text.count("<search>") >=1
        
    ])

def compute_score(solution_str, ground_truth, method="strict", format_score=0.1, score=1.0):
    answer = extract_solution(solution_str=solution_str)
    
    
    # Print only when random sampling is enabled to avoid log overflow
    if random.randint(1, 64) == 1:
        print("---------------start-----------------")
        print(f"Golden answers: {ground_truth.get('target')}")
        print(f"Extracted answer: {answer}")
        print("---------------end-----------------")

    # 1. If no answer was extracted at all
    if answer is None:
        return -0.1

    if len(solution_str) < 10: # Thought process is too short
        return 0  
    
    # 2. Calculate content score 
    final_score = em_check(answer, ground_truth.get("target", []), score)
    
    if final_score > 0:
        # Correct answer (or partially correct), but wrong format -> downgrade penalty
        if not correct_format(solution_str):
            return final_score / 4
        return final_score
    else:
        # Wrong answer, but completely correct format -> give a small encouraging score
        if correct_format(solution_str):
            return format_score
        return 0