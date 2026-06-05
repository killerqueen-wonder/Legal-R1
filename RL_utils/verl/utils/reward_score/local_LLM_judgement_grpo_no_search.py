import re
import random
from openai import OpenAI
import time
# for no search, remove retrieval-related scoring
# =============================================================================
# 1. Core Prompt Definition (Customized for GRPO)
# =============================================================================
# Modified to 0-100 scale, forcing the Judge to give fine-grained scores, expanding intra-group variance for GRPO
UNIFIED_JUDGE_PROMPT_TEMPLATE = """你是一位公正的法律专家评委。请评估AI助手对用户法律问题的回答质量。

[问题]
{question}

[参考轨迹 (Ground Truth CoT)]
{reference_cot}

[参考答案 (Ground Truth Answer)]
{reference_answer}

[被测模型完整输出]
{model_output}

请按照以下维度对AI助手的回答进行综合打分（0 - 100分）：

1. **核心事实覆盖 (Accuracy, 40分)**: AI的最终结论 (<answer>) 是否涵盖了参考答案中的关键法律事实或结论？
2. **检索支撑性 (Grounding, 30分)**: 如果问题复杂到需要检索，AI是否进行了检索?结论是否建立在检索结果之上？
3. **逻辑与完整性 (Reasoning, 30分)**: AI的推理过程是否逻辑严密，是否避免了捏造法条（幻觉）？

**评分参考**：
- **0-30分**: 严重错误。结论与参考答案矛盾，或存在严重幻觉（捏造法条），或未进行检索直接瞎编。
- **31-60分**: 结论基本正确，但检索过程无效，或逻辑混乱，或遗漏关键事实。
- **61-85分**: 结论正确，检索有效，逻辑清晰。允许回答比参考答案更详细。
- **86-100分**: 完美回答。检索精准，推理深刻，结论完全准确且表述专业。

请仅输出一个 JSON 字典，不要包含任何其他分析过程文字，确保包含 "accuracy",一个键。
例如：{{"accuracy": 85}}  /no_think"""



import json

def extract_answer_content(text: str) -> str:
    """
    Extract the content between the [last] occurrence of <answer> and </answer> in the text.
    """
    # Find all matches
    matches = re.findall(r"<answer>(.*?)(?:</answer>|$)", text, re.DOTALL)
    if matches:
        # Return the last match and strip whitespace
        return matches[-1].strip()
    return ""

def extract_answer_content(text):
    match = re.search(r"<answer>(.*?)(</answer>|$)", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def count_search_actions(text: str) -> int:
    """Count the number of occurrences of the <search> tag in the text"""
    if not text or not isinstance(text, str):
        return 0
    return len(re.findall(r"<search>", text))

def extract_information_blocks(text: str) -> str:
    """Extract all <information> content in the text for the judge's reference"""
    matches = re.findall(r"<information>(.*?)</information>", text, re.DOTALL)
    return "\n---\n".join(matches) if matches else "无检索内容"

def calculate_query_quality_score(solution_str: str) -> float:
    """
    Extract similarity score only for "Legal Retrieval" (法律检索).
    Ignore "Similar Case Retrieval" (类案检索), normalized based on a 0-20.0 scale.
    """
    pattern = r"<information>.*?\[Score:\s*([\d\.]+),\s*Type:\s*法律检索\].*?</information>"
    matches = re.findall(pattern, solution_str, re.DOTALL)
    if not matches:
        return 0.0
    scores_100 = [max(0.0, min(100.0, (float(s) / 20.0) * 100.0)) for s in matches]
    return sum(scores_100) / len(scores_100)

import re
import json

def parse_judge_json(raw_str: str) -> dict:
    """Robust JSON parsing (remove <think> blocks, extract core dictionary, retain symbol fault tolerance)"""
    # 1. Remove <think>...</think> blocks (use re.DOTALL to match multi-line content)
    clean_str = re.sub(r"<think>.*?</think>", "", raw_str, flags=re.DOTALL)
    
    # Clean up potentially remaining isolated tags and leading/trailing whitespace
    clean_str = clean_str.replace("<think>", "").replace("</think>", "").strip()
    
    # 2. Fault-tolerant replacement of Chinese and English punctuation marks
    clean_str = clean_str.replace("，", ",").replace("“", '"').replace("”", '"').replace("：", ":")
    
    # 3. Lock the real JSON dictionary boundary (find the first { and the last })
    start_idx = clean_str.find('{')
    end_idx = clean_str.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
        # Precisely intercept the dictionary part
        clean_str = clean_str[start_idx:end_idx+1]
    else:
        # Structural patching: Automatically complete missing braces at the beginning and end (fallback)
        if not clean_str.startswith("{"): clean_str = "{" + clean_str
        if not clean_str.endswith("}"): clean_str = clean_str + "}"
        
    # 4. Load validation
    try:
        data = json.loads(clean_str)
        # Ensure essential keys exist
        if all(k in data for k in ["accuracy"]):
            return data
    except Exception as e:
        # If needed, uncomment here to print the specific parsing error
        print(f"[JSON Decode Error] {e} -> String: {clean_str}")
        return None
        
    return None
    
# =============================================================================
# 2. Singleton Client Management (Synchronous Mode)
# =============================================================================
_GLOBAL_CLIENT = None

def get_client(api_url="http://localhost:9000/v1"):
    global _GLOBAL_CLIENT
    if _GLOBAL_CLIENT is None:
        # Use synchronous calls to avoid event loop conflicts in the RL framework
        _GLOBAL_CLIENT = OpenAI(api_key="EMPTY", base_url=api_url)
    return _GLOBAL_CLIENT



class VLLMRewardManager:
    def __init__(self, api_url="http://localhost:9000/v1"):
        self.client = get_client(api_url)
        self.model_name = "qwen3-8b-reward" 

    def get_unified_subjective_scores(self, question: str, reference_cot: str, reference_answer: str, model_output: str, retrieved_info: str) -> dict:
        """
        Call LLM to get the 3D scoring JSON, retry up to 3 times, return a 0-score dictionary if timeout or error.
        """
        prompt = UNIFIED_JUDGE_PROMPT_TEMPLATE.format(
            question=question,
            reference_cot=reference_cot,
            reference_answer=reference_answer,
            model_output=model_output,
        )
        
        default_scores = {"accuracy": 0.0, "alignment": 0.0, "info_gain": 0.0}
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,  # Give a tiny bit of temperature to prevent always sampling the same dead-loop syntax error
                    max_tokens=64,
                    timeout=30.0      # Set timeout mechanism
                )
                content = response.choices[0].message.content.strip()
                parsed_data = parse_judge_json(content)
                
                if parsed_data is not None:
                    return {
                        "accuracy": float(parsed_data["accuracy"]),
                        # "alignment": float(parsed_data["alignment"]),
                        # "info_gain": float(parsed_data["info_gain"])
                        "alignment": 0,
                        "info_gain": 0
                    }
                else:
                    # Fix point: Explicitly catch parsing failures, and print the original text for Debugging
                    print(f"[Warning] Judge JSON parsing failed (Attempt {attempt+1}/{max_retries}). Original model output: {content}")
                    time.sleep(1)
                    continue
                    
            except Exception as e:
                # Includes timeout, API connection error, parsing exception, etc.
                print(f"[Warning] Judge API error or parsing failed (Attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(1) # Retry after a short wait
                continue
                
        print("[Error] Judge API failed all 3 attempts, giving a 0 score penalty.")
        return default_scores

# =============================================================================
# 3. Auxiliary Tools: More reasonable format checking
# =============================================================================
def correct_format(text):
    """
    Only check if the final output format is standardized. Do not force <search>, allow the model to autonomously decide whether to retrieve.
    """
    if "<answer>" not in text or "</answer>" not in text:
        return False
        
    if "<search>" in text and "</search>" not in text:
        return False
    
    if "<syllogism>" in text and "</syllogism>" not in text:
        return False

    return True



def count_search_actions(text: str) -> int:
    """
    Count the number of occurrences of the <search> tag in the text to measure the model's retrieval frequency.
    """
    if not text or not isinstance(text, str):
        return 0
    return len(re.findall(r"<search>", text))



def compute_score(solution_str, ground_truth, extra_info=None):
    """
    Final Reward function implementation:
    1. Subjective 4D weighting (Acc 0.45, Align 0.20, Info 0.20, Query 0.15)
    2. Search frequency alignment bonus (search_bonus)
    3. Length bonus (length_bonus, max limit 0.1)
    """
    ground_truth = ground_truth.get("target", []) if isinstance(ground_truth, dict) else ground_truth
    if isinstance(ground_truth, list):
        reference = "\n".join([str(x) for x in ground_truth if x])
    else:
        reference = str(ground_truth)
    # Get reference CoT (including thought, search, information, etc.)
    reference_cot = reference
    # Extract reference answer: use extract_answer_content to get the last <answer>
    reference_answer = extract_answer_content(reference_cot)
    
    
    question = extra_info.get('question', "题目缺失") if extra_info else "题目缺失"

    # --- Step 1: Basic format and content gating ---
    if not correct_format(solution_str):
        return 0.0  
    
    answer_content = extract_answer_content(solution_str)
    if len(answer_content) < 3:
        return 0.05  

    # --- Step 2: Search query quality (legal retrieval similarity) ---
    # query_quality_100 = calculate_query_quality_score(solution_str)
    query_quality_100 = 0

    # --- Step 3: LLM Judge multi-dimensional scoring (with retry and parsing) ---
    retrieved_info = extract_information_blocks(solution_str)
    manager = VLLMRewardManager()
    subjective_scores = manager.get_unified_subjective_scores(
        question=question,
        reference_cot=reference_cot,
        reference_answer=reference_answer,
        model_output=solution_str,
        retrieved_info=retrieved_info
    )

    acc_100 = subjective_scores["accuracy"]
    align_100 = subjective_scores["alignment"]
    info_100 = subjective_scores["info_gain"]

    # --- Step 4: Heuristic bonus calculation (search_bonus & length_bonus) ---
    # Search count alignment bonus
    gt_search_count = count_search_actions(reference_cot)
    model_search_count = count_search_actions(solution_str)
    diff = abs(gt_search_count - model_search_count)
    
    search_bonus = 0.0
    if diff <= 1:
        search_bonus = 0.10  
    elif diff <= 2:
        search_bonus = 0.05
        
    # Length bonus item (new formula: max limit 0.1)
    length_punish = min(0.1, len(solution_str) * 0.000001)

    # --- Step 5: Final Score Aggregation ---
    # Ratio: Acc(0.45) + Align(0.20) + Info(0.20) + Query(0.15)
    subjective_total = (acc_100 * 0.8 / 100.0) + \
                       (align_100 * 0.08 / 100.0) + \
                       (info_100 * 0.10 / 100.0) + \
                       (query_quality_100 * 0.02 / 100.0)

    final_score = subjective_total + search_bonus - length_punish

    # --- Log Sampling ---
    if random.randint(1, 64) == 1:
        print(f"\n[GRPO RL Reward] Final: {final_score:.4f}")
        print(f"Components -> Acc:{acc_100}, Align:{align_100}, Info:{info_100}, Query:{query_quality_100:.1f}")
        print(f"Bonuses    -> SearchBonus:{search_bonus}, LengthBonus:{length_punish:.4f}")
        print(f"Q: ...{question[-100:]}")
        print(f"GT: ...{ground_truth[-200:]}")
        print(f"Model think: {solution_str[2000:]}...")
        print(f"Model Answer: {answer_content}...")
        

    return final_score