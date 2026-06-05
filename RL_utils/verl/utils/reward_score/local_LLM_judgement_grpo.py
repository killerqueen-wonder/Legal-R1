import re
import json
import time
import random
import concurrent.futures
from openai import OpenAI, BadRequestError

# =============================================================================
# 1. Split Prompt Definition (Precisely crop context to reduce input length)
# =============================================================================

JUDGE_PROMPT_ACCURACY = """你是一位严谨的法律事实审计员。请对比【参考答案】和【被测模型最终回答】，给出核心事实准确度打分 (1-4分)。

[参考答案]
{reference_answer}
[参考答案结束]

[被测模型最终回答]
{answer_content}
[被测模型最终回答结束]

评分标准：
- [4分]: 最终结论完全准确，与参考答案关键事实一致。
- [3分]: 结论基本正确，但遗漏部分细节或带微小错误。
- [2分]: 结论部分错误。
- [1分]: 严重错误，结论矛盾，回答为空或出现无意义字符。

请仅输出一个评分数字，格式严格为：[[分数]]，例如 [[3]]。不要输出其他解释。 /no_think"""


JUDGE_PROMPT_ALIGNMENT = """你是一位AI推理行为审计员。请对比【参考轨迹】和【被测模型完整轨迹】，根据被测轨迹的检索时机与参考轨迹是否对齐，给出打分 (1-4分)。

[参考轨迹]
{reference_cot}
[参考轨迹结束]

[被测模型完整轨迹]
{model_output}
[被测模型完整轨迹结束]

评分标准：
- [4分]: 检索时机和策略与参考轨迹高度一致。
- [3分]: 检索时机或策略略有偏差，但整体合理。
- [2分]: 错过必要检索节点，或进行了多余的低效检索。
- [1分]: 该检索却未检索，或极度滥用工具。

请仅输出一个评分数字，格式严格为：[[分数]]，例如 [[2]]。不要输出其他解释。 /no_think"""


JUDGE_PROMPT_INFO_GAIN = """你是一位信息价值审计员。请对比【被测模型的检索内容】和【参考答案】，判断被测模型的检索内容是否与参考答案相关，给出信息增量价值打分 (1-4分)。

[参考答案]
{reference_answer}
[参考答案结束]

[被测模型的检索内容]
{retrieved_info}
[被测模型的检索内容结束]

评分标准：
- [4分]: 搜回信息极精准，直接支撑参考答案的核心内容。
- [3分]: 信息部分相关，提供背景支持但非关键一击。
- [2分]: 搜到多为边缘信息，帮助有限。
- [1分]: 完全无关，或未进行有效检索(无返回)。

请仅输出一个评分数字，格式严格为：[[分数]]，例如 [[3]]。不要输出其他解释。 /no_think"""

JUDGE_PROMPT_TEMPLATE = """你是一位严谨的 AI 行为与法律事实审计员。请根据提供的【问题】、【参考轨迹】、【参考答案】，对【被测模型完整轨迹】、【被测模型检索内容】、【被测模型最终回答】进行三个维度的严格打分 (1-4分)。
【问题】是用户提出的问题，【参考轨迹】是参考模型对问题的COT过程，【参考答案】是参考模型给出的答案，【被测模型完整轨迹】是被测模型对问题的COT过程，【被测模型检索内容】是被测模型在思考问题时检索的内容，【被测模型最终回答】是被测模型给出的答案。
你需要针对被测模型的三项输出内容，分别给出三项分数。以下提供所有信息，以及三项评分标准。
=== 输入信息 ===
[问题]
{question}

[参考轨迹]
{reference_cot}

[参考答案]
{reference_answer}

[被测模型完整轨迹]
{model_output}

[被测模型检索内容]
{retrieved_info}

[被测模型最终回答]
{answer_content}
=== 信息结束 ===

=== 评分标准 ===
1. 核心事实准确度 (accuracy): 对比【参考答案】和【被测模型最终回答】。
   - [4分]: 最终结论完全准确，与参考答案关键事实一致。
   - [3分]: 结论基本正确，但遗漏部分细节或带微小错误。
   - [2分]: 结论部分错误。
   - [1分]: 严重错误，结论矛盾，回答为空或出现无意义字符。

2. 检索策略与对齐度 (alignment): 对比【参考轨迹】和【被测模型完整轨迹】。
   - [4分]: 检索时机和策略与参考轨迹高度一致。
   - [3分]: 检索时机或策略略有偏差，但整体合理。
   - [2分]: 错过必要检索节点，或进行了多余的低效检索。
   - [1分]: 该检索却未检索，或极度滥用工具。

3. 信息增量价值 (info_gain): 对比【被测模型的检索内容】和【参考答案】。
   - [4分]: 搜回信息极精准，直接支撑参考答案的核心内容。
   - [3分]: 信息部分相关，提供背景支持但非关键一击。
   - [2分]: 搜到多为边缘信息，帮助有限。
   - [1分]: 完全无关，或未进行有效检索(无返回)。

=== 输出要求 ===
请仔细思考以上三个维度的表现，最后**仅输出**一个合法的 JSON 对象，不要输出任何其他的解释文字、Markdown 代码块或思考过程标记。格式必须严格如下：
{{"accuracy": 0, "alignment": 0, "info_gain": 0}}
请输出你的评分："""

# =============================================================================
# 2. Helper Functions (Keep unchanged)
# =============================================================================
def extract_answer_content(text):
    match = re.search(r".*<answer>(.*?)</answer>", text, re.DOTALL)
    if match: return match.group(1).strip()
    last_index = text.rfind("<answer>")
    if last_index != -1: return text[last_index + len("<answer>"):].strip()
    return ""

def count_search_actions(text: str) -> int:
    return len(re.findall(r"<search>", str(text)))

def extract_information_blocks(text: str) -> str:
    matches = re.findall(r"<information>(.*?)</information>", str(text), re.DOTALL)
    return "\n---\n".join(matches) if matches else "无检索内容"

def calculate_query_quality_score(solution_str: str) -> float:
    pattern = r"<information>.*?\[Score:\s*([\d\.]+),\s*Type:\s*法律检索\].*?</information>"
    matches = re.findall(pattern, str(solution_str), re.DOTALL)
    if not matches: return 0.0
    scores_100 = [max(0.0, min(100.0, (float(s) / 20.0) * 100.0)) for s in matches]
    return sum(scores_100) / len(scores_100)


def parse_scores(text: str) -> dict:
    """
    # Highly robust score extraction function:
    # 1. Read from the first { to the first }
    # 2. Remove spaces, change all Chinese quotes, colons, and commas to English
    # 3. Try JSON parsing
    # 4. If failed, use keyword regex matching, record unmatched items as 0.
    """
    default_scores = {"accuracy": 0, "alignment": 0, "info_gain": 0}
    if not text:
        return default_scores

    start = text.find('{')
    end = text.find('}')
    
    if start != -1 and end != -1 and end > start:
        # 1. Intercept dictionary string
        json_str = text[start:end+1]
        
        # 2. Clean: Remove whitespace characters (including spaces and line breaks)
        json_str = re.sub(r'\s+', '', json_str)
        # 3. Clean: Chinese punctuation replacement
        json_str = json_str.replace("“", '"').replace("”", '"').replace("：", ":").replace("，", ",")
        
        try:
            # 4. Try parsing as valid JSON
            data = json.loads(json_str)
            return {
                "accuracy": int(data.get("accuracy", 0)),
                "alignment": int(data.get("alignment", 0)),
                "info_gain": int(data.get("info_gain", 0))
            }
        except Exception:
            pass # If exceptions like JSONDecodeError are thrown, pass to the regex fallback below

    # 5. Regex Fallback matching
    # Match pattern explanation: find keywords, allow any non-digit characters in between, then capture the first appearing digit
    for key in default_scores.keys():
        # Try to match quoted "accuracy": 3
        match = re.search(rf'"{key}"[^0-9]*([0-9])', text, re.IGNORECASE)
        if not match:
            # Try to match unquoted accuracy: 3
            match = re.search(rf'{key}[^0-9]*([0-9])', text, re.IGNORECASE)
        
        if match:
            default_scores[key] = int(match.group(1))

    return default_scores

def correct_format(text):
    # answer must be complete, search and syllogism must be closed
    
    # Find all <answer> and </answer> tags in order
    tags = re.findall(r'(<answer>|</answer>)', text)
    
    # 1. If the total number of tags is less than 2, or not an even number (appearing in pairs), directly return False
    if len(tags) < 2 or len(tags) % 2 != 0:
        return False
    
    # 2. Check if they appear alternately
    # Even indices (0, 2, 4...) must be '<answer>'
    # Odd indices (1, 3, 5...) must be '</answer>'
    for i, tag in enumerate(tags):
        if i % 2 == 0 and tag != '<answer>':
            return False
        if i % 1 == 0 and i % 2 != 0 and tag != '</answer>':
            return False
        
    if "<search>" in text and "</search>" not in text: return False
    if "<syllogism>" in text and "</syllogism>" not in text: return False
    return True

def check_information_tags_strict(text: str) -> bool:
    """
    Determine whether there is at least one pair of <information> tags in the string,
    and <information> and </information> must appear strictly alternately.
    """
    # Use regular expressions to extract all start and end tags, and store them in a list in order of appearance
    # </?information> will match <information> and </information>
    tags = re.findall(r'</?information>', text)
    
    # Condition 1: Tags must exist at least (and if alternating correctly, the total must be an even number, at least 2)
    if not tags:
        return False
        
    # State variable: 0 indicates waiting for start tag, 1 indicates waiting for end tag
    state = 0 
    
    for tag in tags:
        if tag == "<information>":
            if state == 1:
                # Error: The previous tag was also a start tag (e.g., <info> <info>)
                return False
            state = 1
            
        elif tag == "</information>":
            if state == 0:
                # Error: Ended directly without a start tag, or consecutive end tags appeared
                return False
            state = 0
            
    # Condition 2: After traversal, the state must return to 0, meaning all opened tags have correctly closed
    return state == 0

def check_search_json(text: str) -> bool:
    """
    Check if there is at least one pair of <search> and </search> in the string,
    and it contains JSON text that meets specific format requirements.
    """
    # Use regular expressions to extract all content between <search> and </search>
    # The re.DOTALL parameter allows '.' to match any character including newlines
    matches = re.findall(r'<search>(.*?)</search>', text, re.DOTALL)
    
    for content in matches:
        try:
            # Try to parse the extracted content into a JSON dictionary
            # json.loads will automatically ignore leading and trailing whitespace and newlines
            data = json.loads(content)
            
            # Ensure the parsed result is a JSON object (dictionary), not an array or a normal string
            if not isinstance(data, dict):
                continue
            
            # Extract all JSON keys into a set for exact comparison
            keys = set(data.keys())
            
            # Validation Rule 1: Legal Search
            if data.get("检索类型") == "法律检索":
                expected_keys = {"检索类型", "关键词", "检索目的"}
                # Set comparison: not only must these three keys be included, but there cannot be other extra keys
                if keys == expected_keys:
                    return True
                    
            # Validation Rule 2: Similar Case Search
            elif data.get("检索类型") == "类案检索":
                expected_keys = {"检索类型", "检索案情", "罪名", "其他情节"}
                # Set comparison: must and can only contain these four keys
                if keys == expected_keys:
                    return True
                    
        except json.JSONDecodeError:
            # If json.loads throws an exception, it means the content inside the tags is not valid JSON
            # Catch the exception and skip, continue to check the next pair of tags
            continue
            
    # If no matching items meet the conditions after traversing all, return False
    return False

# =============================================================================
# 3. Client and Request Management (Refactored version)
# =============================================================================
_GLOBAL_CLIENT = None

def get_client(api_url="http://localhost:9000/v1"):
    global _GLOBAL_CLIENT
    if _GLOBAL_CLIENT is None:
        _GLOBAL_CLIENT = OpenAI(api_key="EMPTY", base_url=api_url)
    return _GLOBAL_CLIENT

class VLLMRewardManager:
    def __init__(self, api_url="http://localhost:9000/v1"):
        self.client = get_client(api_url)
        self.model_name = "qwen3-8b-reward" 

    def get_unified_subjective_scores(self, question: str, reference_cot: str, reference_answer: str, retrieved_info: str, answer_content: str, model_output: str) -> dict:
        """Obtain scores of three dimensions in a single API call, handle over-length truncation, retries, and exception masking"""
        max_retries = 3
        
        # Initialize input parameter dictionary
        kwargs = {
            "question": question,
            "reference_cot": reference_cot,
            "reference_answer": reference_answer,
            "model_output": model_output,
            "retrieved_info": retrieved_info,
            "answer_content": answer_content
        }
        
        for attempt in range(max_retries):
            prompt = JUDGE_PROMPT_TEMPLATE.format(**kwargs)
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=128, # Relax token limit since full JSON format output is required
                    timeout=300.0 
                )
                content = response.choices[0].message.content.strip()
                scores = parse_scores(content)
                return scores
                
            except BadRequestError as e:
                # Catch over-length error: HTTP 400
                error_msg = str(e).lower()
                if "maximum context length" in error_msg or "context" in error_msg:
                    print(f"[Warning] Unified Scoring API triggered Context Limit (Attempt {attempt+1}/{max_retries}). Trying to truncate text...")
                    
                    # Combined total length becomes larger, single field truncation length should be more conservative, keep about 15000 chars (~10000 tokens)
                    max_chars = 15000 
                    needs_retry = False
                    if len(kwargs["model_output"]) > max_chars:
                        kwargs["model_output"] = "...[前文已截断]..." + kwargs["model_output"][-max_chars:]
                        needs_retry = True
                    if len(kwargs["reference_cot"]) > max_chars:
                        kwargs["reference_cot"] = "...[前文已截断]..." + kwargs["reference_cot"][-max_chars:]
                        needs_retry = True
                        
                    if not needs_retry:
                        break # If no long fields to truncate, break retry directly
                    continue 
                else:
                    print(f"[Warning] Unified Scoring API encountered other 400 errors: {e}")
                    time.sleep(1)
                    continue
                    
            except Exception as e:
                # Other API exceptions (connection errors, timeouts, etc.)
                print(f"[Warning] Unified Scoring API unknown exception (Attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(1)
                
        print(f"[Error] Unified Scoring API final fetch failed, returning all 0 scores.")
        return {"accuracy": 0, "alignment": 0, "info_gain": 0}
    
# =============================================================================
# 4. Core Scoring Logic 
# =============================================================================
def compute_score(solution_str, ground_truth, extra_info=None):
    
    ground_truth = ground_truth.get("target", []) if isinstance(ground_truth, dict) else ground_truth
    if isinstance(ground_truth, list):
        reference = "\n".join([str(x) for x in ground_truth if x])
    else:
        reference = str(ground_truth)
        
    reference_cot = reference
    reference_answer = extract_answer_content(reference_cot)
    question = extra_info.get('question', "题目缺失") if extra_info else "题目缺失"

    final_score = 0
    # --- Step 1: Basic format and content gating ---
    if correct_format(solution_str):
        final_score += 0.50  # Replaces the original -0.2

    if check_search_json(solution_str):
        final_score += 0.25  # Originally +0.1

    if check_information_tags_strict(solution_str):
        final_score += 0.25  # Originally +0.1
    
    answer_content = extract_answer_content(solution_str)
    

    # --- Step 2: Query quality ---
    query_quality_100 = calculate_query_quality_score(solution_str)

    # --- Step 3: LLM Judge multi-dimensional scoring (merged retrieval) ---
    retrieved_info = extract_information_blocks(solution_str)
    manager = VLLMRewardManager()
    
    # Pass all required parameters
    subjective_scores = manager.get_unified_subjective_scores(
        question=question,
        reference_cot=reference_cot,
        reference_answer=reference_answer,
        retrieved_info=retrieved_info,
        answer_content=answer_content,
        model_output=solution_str
    )

    acc_4 = subjective_scores["accuracy"]
    align_4 = subjective_scores["alignment"]
    info_4 = subjective_scores["info_gain"]

    # --- Step 4: Heuristic bonus calculation ---
    gt_search_count = count_search_actions(reference_cot)
    model_search_count = count_search_actions(solution_str)
    diff = abs(gt_search_count - model_search_count)
    
    search_bonus = 0.0
    # Number of searches is close to the reference trajectory
    if diff <= 1: search_bonus = 0.1  
    # Simple question, neither reference nor tested trajectory needs retrieval, compensate for possible loss of retrieval info score
    if gt_search_count == 0 and model_search_count==0 : search_bonus = 0.2
        
    length_punish_limit = 0.6
    length_punish = min(length_punish_limit, (len(solution_str)*0.000005)*length_punish_limit)

    if answer_content=='和' or answer_content=='':
        # No output answer, directly judge as negative
        acc_4 = 0.1

    # --- Step 5: Final score aggregation ---
    subjective_total = (acc_4 * 1.00 / 4.0) + \
                       (align_4 * 0.08 / 4.0) + \
                       (info_4 * 0.32 / 4.0) + \
                       (query_quality_100 * 0.20 / 100.0)
    
    

    final_score = final_score + subjective_total + search_bonus - length_punish

    if random.randint(1, 32) == 1:
        print("=="*20)
        print(f"\n[GRPO RL Reward] Final: {final_score:.4f}")
        print(f"Components -> Acc:{acc_4}, Align:{align_4}, Info:{info_4}, Query:{query_quality_100:.1f}")
        print(f"Bonuses    -> search_bonus:{search_bonus:.4f},LengthBonus:{length_punish:.4f}")
        print(f"Q: ...{question[-200:]}")
        print(f"GT: ...{reference[-1000:]}")
        print(f"Model think: {solution_str}...")
        print(f"Model Answer: {answer_content}...")
        print("=="*20)
         
    return final_score