import os
import re
import json
import time
import requests
import pandas as pd
import subprocess
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor
from vllm import LLM, SamplingParams


# ================= Configuration Area =================
# Update to your JSON dataset path
DATASET_PATH = "./data/train_data/RL/dataset.json"
MODEL_PATH = "/path/to/your/model"
RERANK_MODEL_PATH = "/path/to/your/model"

OUTPUT_FILE = "/path/to/your/rollout_trajectories.jsonl"

SAMPLE_SIZE = 3000  # Final sampled data size
MAX_TURNS = 9
NUM_GPUS = 2
SEARCH_URL = "http://127.0.0.1:8005/retrieve"
SEARCH_TOPK = 8
FILTER_VLLM_PORT = 8007

# 1. Filter vLLM Start Command (Provides filtering compute for RAG)
FILTER_VLLM_START_COMMAND = f"""
export CUDA_VISIBLE_DEVICES=3
source /path/to/your/conda.sh
conda activate vllm_server
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python -m vllm.entrypoints.openai.api_server \\
    --model {RERANK_MODEL_PATH} \\
    --served-model-name Qwen3-8B \\
    --port {FILTER_VLLM_PORT} \\
    --enable-chunked-prefill \\
    --max-model-len 12000 \\
    --gpu-memory-utilization 0.40 \\
    --max-num-seqs 64 \\
    --dtype bfloat16 \\
    --trust-remote-code
"""

RAG_START_COMMAND = f"""
export CUDA_VISIBLE_DEVICES=3
source /path/to/your/conda.sh
conda activate retriever_filter
cd ./RL_utils
export TRANSFORMERS_CACHE=/path/to/your/cache_dir
export HF_HUB_CACHE=/path/to/your/cache_dir
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
python search_r1/search/async_retrieval_server.py \
    --port 8005 \
    --corpus_path './data/RAG/legal_corpus.jsonl' \
    --case_corpus_path './data/RAG/Criminal Precedent Corpus_psi_subset.jsonl' \
    --retriever_name hybrid_filter \
    --dictionary_path './data/RAG/dictionary.txt' \
    --search_depth 5 --bm25_weight 15 --bm25_weight_factor 2 --bm25_k1 0.15 --bm25_b 0.35 --topk 8 \
    --retriever_model 'shibing624/text2vec-base-chinese-paraphrase' \
    --vllm_url http://127.0.0.1:{FILTER_VLLM_PORT}/v1/completions \
    --gpu_ids 3 --gpu_memory_limit_per_gpu 5
"""

# ================= Custom Prompt Wrapper =================
def make_prefix(question_text):
    """Generate Prompt with think-retrieve-answer instructions"""
    query_prompt_init = """
你是一个严谨且专业的法律AI助手。你的任务是通过逐步思考用户请求并回答法律问题。回答必须基于事实，严禁编造法律条文或案例。

### 核心指令：
0.**第一步思考**: 识别用户请求的核心法律实体，关键事实，并且提出可能涉及的法律条文。

1. **判断是否需要检索**：你可以使用检索工具。如果问题基础且你非常有把握，也可以不使用工具直接作答。


2. **支持的检索工具**（两种）：
   - **法律检索**：需要确认某项罪名的具体刑期、适用条件、或者某一司法解释的原文时使用。
   - 不要试图一次性把所有关键词都搜完。每次最多只查找两项最相关的法律。
   - 先搜索最核心的概念。不要用缩写词，尽量用完整的，最有特点的，区别于其他法条的关键词。搜索关键词示例：“刑法 盗窃罪”、“最高人民法院关于适用〈民事诉讼法〉的解释 第501条”。
   - 如果你搜索了三次依然没有找到相关条文，请直接承认未找到，修改思考思路，搜索其他条文。不要编造内容。
   - **类案检索**：用来检索相似刑事案件的判例报告（案例库只有刑事），以预测判决结果或量刑，提高置信度。

3. **如何调用工具**：如果你决定检索，**必须**输出一个严格的 JSON 字符串，并用 `<search>` 和 `</search>` 标签包裹。
   - **调用【法律检索】的 JSON 格式示例**：
     <search>
     {{
       "检索类型": "法律检索",
       "关键词": "刑法 第xxx条 盗窃罪（你想要找的法条的编号和法条原文中包含的关键词）",
       "检索目的": "找到刑法中盗窃罪的刑期判定条文"
     }}
     </search>
   - **调用【类案检索】的 JSON 格式示例**：
     <search>
     {{
       "检索类型": "类案检索",
       "检索案情": "张三蒙面进入邻居家，偷走现金5000元并持刀威胁屋主。",
       "罪名": ["盗窃罪", "抢劫罪"],
       "其他情节": "自首悔过"
     }}
     </search>

4. **三段论推理**：
   在你接收到【法律检索结果】后，如果用户的案例事实与某条检索到的法律法规能够匹配，你**必须**在接下来的思考中，首先使用 `<syllogism>` 标签生成一个三段论 JSON 进行法理分析。
   - **大前提 (Major Premise)**：指代适用的具体罪名或法条。**注意：绝对不要重复输出法条原文，必须严格使用占位符 `[法条参考 X]`**（X为检索结果给出的序号，例如 `[法条参考 1]`）。
   - **小前提 (Minor Premise)**：将用户平常的语言表述转化为专业的**法言法语**。
   - **结论 (Conclusion)**：案件事实是否符合该法条，当事人是否适用该法条，以及根据法条应该如何定罪或量刑。
   
   **三段论 JSON 格式示例**：
   <syllogism>
   {{
     "Major Premise": "刑法第264条 盗窃罪 [法条参考 1]",
     "Minor Premise": "张三于某日以非法占有为目的，入室秘密窃取他人财物，共计金额5000元...",
     "Conclusion": "张三的行为符合盗窃罪的构成要件，适用该法条，判处..."
   }}
   </syllogism>
   *(注意：如果检索结果无法匹配事实，则不要生成该三段论标签和内容)*

5. **多轮迭代检索**：每次只输出一个 `<search>` 标签。接收结果后分析是否充足。

6. **最终回答**：收集充分后，必须将最终推理结论包裹在 `<answer>` 和 `</answer>` 标签中。
### 回答流程：
- 遇到问题 -> 分析问题复杂度，发现不需要查询资料 -> 思考问题 -> 回答
- 遇到问题 -> 分析问题复杂度，发现需要查询资料 -> **调用工具** (此时你会暂停) -> 接收工具结果 -> 分析结果 -> 发现还需要查别的 -> **再次调用工具** ... -> 最终整合信息回答。
以下是需要回答的问题：{question_text}\n
"""
    return query_prompt_init.format(question_text=question_text)

# ================= RAG Lifecycle Management =================
def wait_for_port(port: int, host: str = '127.0.0.1', timeout: int = 1200):
    """Wait for local port to be ready"""
    import socket
    start_time = time.time()
    print(f"[INFO] Waiting for RAG service to start (port {port})...")
    while time.time() - start_time < timeout:
        try:
            with socket.create_connection((host, port), timeout=1):
                print(f"[INFO] RAG service port {port} is ready!")
                return True
        except OSError:
            time.sleep(5)
    raise RuntimeError(f"RAG service failed to start within {timeout} seconds.")

# ================= RAG Request Logic =================
def local_rag_search(search_json_str: str) -> str:
    """Clean and parse JSON, request local RAG API, and format results"""
    try:
        
        clean_str = search_json_str.strip()
        # 1. Strip potential Markdown code blocks (e.g., ```json and ```)
        if clean_str.startswith("```json"):
            clean_str = clean_str[7:]
        elif clean_str.startswith("```"):
            clean_str = clean_str[3:]
        if clean_str.endswith("```"):
            clean_str = clean_str[:-3]
            
        clean_str = clean_str.strip()
        
        # 2. Fault-tolerant replacement for Chinese punctuation
        clean_str = clean_str.replace("，", ",")        # Chinese comma to English comma
        clean_str = clean_str.replace("“", '"')       # Chinese left double quote to English
        clean_str = clean_str.replace("”", '"')       # Chinese right double quote to English
        
        # 3. Structural repair: auto-complete missing curly braces at start/end
        if not clean_str.startswith("{"):
            clean_str = "{" + clean_str
        if not clean_str.endswith("}"):
            clean_str = clean_str + "}"
        # ================= End of JSON Robustness Cleaning =================
        search_query = json.loads(clean_str)
        
        payload = {"query": search_query, "topk": SEARCH_TOPK}
        
        response = requests.post(SEARCH_URL, json=payload, timeout=3000.0)
        response.raise_for_status()
        json_data = response.json()
        
        if "error" in json_data:
            return f"检索返回错误：{json_data['error']}"

        req_type = json_data.get("检索类型", "")
        if req_type == "类案检索":
            summary = json_data.get("llm_summary", "未检索到匹配的类案分析结果。")
            return f"【类案检索分析报告】\n{summary}"
        elif req_type == "法律检索":
            results = json_data.get("result", [])
            format_reference = [f"法条参考 {idx + 1} (相关度: {doc.get('score', 0.0):.4f}):\n{doc.get('document', {}).get('content', '')}\n" 
                                for idx, doc in enumerate(results)]
            if not format_reference:
                return "【法律检索结果】未找到相关的法律条文，请尝试更换关键词。"
            return "【法律检索结果】\n" + "\n".join(format_reference)
        else:
            return f"未知的检索类型返回，原始数据: {str(json_data)[:200]}"
            
    except json.JSONDecodeError:
        return "工具调用失败：<search>标签内的JSON格式不合法，请检查并输出合法的JSON格式再试一次。"
    except Exception as e:
        return f"检索请求出错: {str(e)}"
    except NameError:
        return "代码逻辑错误：clean_str 未定义。请在脚本中补充 JSON 清洗逻辑。"

# ================= Main Control Flow =================
def main():

    print(f"[INFO] Starting Filter vLLM subprocess (port {FILTER_VLLM_PORT})...")
    filter_process = subprocess.Popen(FILTER_VLLM_START_COMMAND, shell=True, executable='/bin/bash')
    wait_for_port(FILTER_VLLM_PORT)
    
    # 1. Start and wait for RAG service
    print("[INFO] Starting RAG subprocess...")
    rag_process = subprocess.Popen(RAG_START_COMMAND, shell=True, executable='/bin/bash')
    wait_for_port(8005)

    # 2. Read JSON dataset and perform "stratified sampling with guaranteed inclusion"
    print(f"[INFO] Loading dataset: {DATASET_PATH}")
    # Assume JSON is a direct list containing dictionaries
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    
    print(f"[INFO] Extracting {SAMPLE_SIZE} samples, ensuring all format_ids are included...")
    # Step 1: Ensure at least 1 sample per format_id
    guaranteed_samples = df.groupby('format_id').sample(n=1, random_state=42)
    
    # Step 2: Calculate remaining needed samples, randomly draw from the remaining pool
    remaining_needed = SAMPLE_SIZE - len(guaranteed_samples)
    remaining_pool = df.drop(guaranteed_samples.index)
    
    if remaining_needed > 0:
        random_samples = remaining_pool.sample(n=remaining_needed, random_state=42)
        df_sampled = pd.concat([guaranteed_samples, random_samples])
    else:
        # Extreme case: total number of categories exceeds target sample size, truncate directly
        df_sampled = guaranteed_samples.head(SAMPLE_SIZE)
        
    # Shuffle the final dataset order
    df_sampled = df_sampled.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Initialize LLM (restricted to GPUs 0, 1)
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
    print(f"[INFO] Initializing vLLM engine on GPUs 0,1: {MODEL_PATH}")
    llm = LLM(
        model=MODEL_PATH, 
        tensor_parallel_size=NUM_GPUS, 
        trust_remote_code=True,
        enable_prefix_caching=True,
        max_model_len=40000, 
        gpu_memory_utilization=0.85 
    )
    
    tokenizer = llm.get_tokenizer()
    
    # Extract and format initial Prompt, keeping specified fields
    trajectories = []
    MAX_PROMPT_LEN = 25000  # Set safety skip threshold
    for idx, row in df_sampled.iterrows():
        # Extract target fields
        data_format_id = row.get('format_id', 'unknown')
        original_question = str(row.get('Open-ended Verifiable Question', ''))
        original_golden_answer = str(row.get('Ground-True Answer', ''))
        reference_cot = str(row.get('Complex_CoT', ''))
        
        # Generate input with prompts using custom function
        user_msg = make_prefix(original_question)
            
        # Wrap using Qwen Chat Template
        formatted_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}], 
            tokenize=False, 
            add_generation_prompt=True
        )

        # ====== New: Length validation and skip logic ======
        token_ids = tokenizer.encode(formatted_prompt)
        
        if len(token_ids) > MAX_PROMPT_LEN:
            print(f"\n[WARNING] Found exceptionally long data! Index: {idx}, Length: {len(token_ids)} tokens, skipping directly.")
            # Print the last 2000 characters of the string to prevent screen flooding
            print(f"[Data tail preview]:\n...{formatted_prompt[-2000:]}")
            print("-" * 60)
            continue  # Skip current iteration, do not add to trajectories
        
        trajectories.append({
            "id": idx,
            "format_id": data_format_id,
            "original_question": original_question,
            "original_golden_answer": original_golden_answer,
            "reference_cot": reference_cot,
            "initial_prompt": user_msg,  # Question with wrapped prompts
            "current_llm_input": formatted_prompt, 
            "turns": [],
            "generated_cot": "",  # Used to concatenate all thought trajectories during generation
            "final_answer": None,
            "is_done": False
        })

    print(f"[INFO] Filtering complete. Total valid data entering Rollout: {len(trajectories)}.")

    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=0.5,
        max_tokens=1200,
        stop=['</search>', '</answer>'],
        include_stop_str_in_output=True 
    )

    # 3. Start Rollout loop
    for turn in range(MAX_TURNS):
        active_indices = [i for i, t in enumerate(trajectories) if not t["is_done"]]
        if not active_indices:
            print("[INFO] All tasks completed, ending early.")
            break
            
        print(f"\n========== Starting generation for turn {turn} (Active tasks: {len(active_indices)}) ==========")
        active_prompts = [trajectories[i]["current_llm_input"] for i in active_indices]
        
        # Batch execute LLM generation
        outputs = llm.generate(active_prompts, sampling_params, use_tqdm=True)

        # Build environmental observations for the current turn (ready to send to RAG)
        turn_rag_queries = []
        turn_mappings = []
        
        for i, output in enumerate(outputs):
            traj_idx = active_indices[i]
            traj = trajectories[traj_idx]
            gen_text = output.outputs[0].text
            
            # Append newly generated text to the generated COT
            traj["generated_cot"] += gen_text
            
            # Parse actions
            pattern = r'<(search|answer)>(.*?)</\1>'
            match = re.search(pattern, gen_text, re.DOTALL)
            
            turn_record = {
                "turn_id": turn,
                "llm_output": gen_text,
                "action": None,
                "search_query": None,
                "rag_result": None
            }
            
            if match:
                action = match.group(1)
                content = match.group(2).strip()
                turn_record["action"] = action
                
                if action == "answer":
                    traj["final_answer"] = content
                    traj["is_done"] = True
                    traj["current_llm_input"] += gen_text
                elif action == "search":
                    turn_record["search_query"] = content
                    turn_rag_queries.append(content)
                    turn_mappings.append((traj_idx, turn_record, gen_text))
            else:
                turn_record["action"] = "invalid"
                traj["current_llm_input"] += gen_text + f'\n【系统提示】 如果下一步需要搜索，应该把搜索内容放在<search> 和 </search>之间。 如果下一步给出最终回答，应该把答案放在 <answer> 和 </answer>之间。让我重新思考。【/系统提示】\n'
                
            traj["turns"].append(turn_record)

        # Batch execute RAG requests and update environment
        if turn_rag_queries:
            print(f"[INFO] Triggering {len(turn_rag_queries)} concurrent RAG retrievals...")
            with ThreadPoolExecutor(max_workers=len(turn_rag_queries)) as executor:
                rag_results = list(executor.map(local_rag_search, turn_rag_queries))
                
            for (traj_idx, turn_record, gen_text), rag_result in zip(turn_mappings, rag_results):
                turn_record["rag_result"] = rag_result
                traj = trajectories[traj_idx]
                
                turn_info = f"\n[系统提示] 当前为第 {turn + 1} 轮检索（上限 {MAX_TURNS} 轮）。"
                if turn + 1 >= MAX_TURNS:
                    turn_info += " 以下是最后一次检索结果。注意：接下来总结以上思考，必须给出最终回答！"
                    
                next_obs = f'\n{turn_info}\n<information>{rag_result.strip()}</information>\n\n'
                traj["current_llm_input"] += gen_text + next_obs
                # Also append retrieved information to generated COT record to maintain full trajectory
                traj["generated_cot"] += next_obs
                
    # 4. Save results
    print(f"\n[INFO] Rollout complete, saving results to {OUTPUT_FILE} ...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for traj in trajectories:
            f.write(json.dumps(traj, ensure_ascii=False) + '\n')
            
    # 5. Clean up subprocesses
    print("[INFO] Terminating RAG service process...")
    rag_process.terminate()
    print("[INFO] Script exited safely.")

if __name__ == "__main__":
    main()