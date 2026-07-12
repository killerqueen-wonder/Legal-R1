import json
import os
import re
import logging
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from retrying import retry
from openai import OpenAI

templates_config=r'YOUR_PATH\\Legal-R1\\data\\train_data_construction\\templates_config.json'
output_path=r'YOUR_OUTPUT_PATH'
#记录跳过id，断点续传
# --- 1. 数据合成专用 Prompt ---
synthesize_system_prompt = """你是一个严谨的法律数据集构建专家。你的任务是根据提供的原始法律文本（如裁判文书）和具体的任务要求，提取并改写出高质量的问答对（QA）用于训练大语言模型。

### 核心指令：
1. **材料适用性判断**：在提取前，请严格判断【参考材料】是否包含满足【任务要求】所需的信息（例如：任务要求提取刑期，但材料是民事纠纷或尚未判决）。如果材料不符合需求、缺失关键信息或无法提取出客观答案，请直接返回带有空值的JSON：{"Question": "", "Ground-True Answer": ""}
2. **问题构建 (Question)**：如果材料符合要求，请根据【任务要求】，从【参考材料】中提取必要的前提信息（如纯粹的案情经过、前科情况等）来构建问题。
3. **答案提取 (Ground-True Answer)**：根据【参考材料】中的实际结果（如判决结果、具体刑期、罪名等）提取出标准答案。答案必须客观、简明扼要。
4. **严禁泄露（最核心要求）**：你构建的问题（Question）文本中，**绝对不能**包含或泄露任何关于答案（Ground-True Answer）的信息。
5. **严格JSON输出**：请直接输出合法的JSON格式，不要包含任何Markdown代码块标记（如```json）或额外的解释说明。

### 输出格式必须为以下两种之一：
符合要求时：
{
    "Question": "结合案情事实描述构建的提问（必须已剔除答案信息）...",
    "Ground-True Answer": "提取出的标准答案（例如：有期徒刑三年）..."
}
不符合要求时：
{
    "Question": "",
    "Ground-True Answer": ""
}
"""

synthesize_user_template = """
【任务要求】
{description}

【参考材料】
{material}

请严格按照系统指令的要求，生成 JSON 格式的 QA 对。
"""

def parse_synthesis_response(response_text: str) -> tuple[bool, Optional[Dict[str, str]]]:
    """清理并解析 LLM 返回的 JSON，处理空值跳过，并增加关键词泄露检测"""
    response = response_text.strip().strip("```").replace("json", "", 1).strip()
    try:
        if not response.startswith('{'):
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                response = match.group(0)
        
        parsed_data = json.loads(response)
        question = parsed_data.get("Question")
        answer = parsed_data.get("Ground-True Answer")

        if question == "" and answer == "":
            logging.getLogger("Parser").info("模型判断：材料不符合任务需求，跳过该材料。")
            return False, None

        return True, parsed_data
    
        if question and answer:
            # 基础硬匹配：检查答案字符串是否直接出现在问题中
            if isinstance(answer, str) and answer in question:
                logging.getLogger("Parser").warning("检测到答案泄露：答案直接出现在问题中。")
                return False, None
            
            # 关键数值检测：如果答案包含刑期数字，检查数字是否出现在问题中
            if isinstance(answer, str):
                duration_match = re.search(r'(\d+|一|二|三|四|五|六|七|八|九|十)[年|个?月|天]', answer)
                if duration_match:
                    duration_val = duration_match.group(0)
                    if duration_val in question:
                        logging.getLogger("Parser").warning(f"检测到疑似泄露：刑期关键词 '{duration_val}' 出现在问题中。")
                        return False, None

            return True, parsed_data
        return False, None
    except Exception as e:
        logging.getLogger("Parser").warning(f"解析失败: {e}")
        return False, None

# --- 2. 数据结构定义 ---
@dataclass
class QAFormat:
    format_id: str
    source_type: str
    source_path: str 
    description: str
    synthesis_num: int = 100
    content_field: str = "contents"
    filter_path: str = ""       
    filter_field: str = ""      

# --- 3. 核心组件 ---
class TemplateManager:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.formats: List[QAFormat] = self._load_templates()

    def _load_templates(self) -> List[QAFormat]:
        with open(self.config_path, 'r', encoding='utf-8') as f:
            raw_formats = json.load(f)
        
        formats_list = []
        for item in raw_formats:
            formats_list.append(QAFormat(
                format_id=item["format_id"],
                source_type=item["source_type"],
                source_path=item["source_path"],
                description=item["description"],
                synthesis_num=item.get("synthesis_num", 100),
                content_field=item.get("content_field", "contents"),
                filter_path=item.get("filter_path", ""),      
                filter_field=item.get("filter_field", "")     
            ))
        return formats_list

class DataLoader:
    def __init__(self):
        self.source_cache: Dict[str, List[Dict[str, Any]]] = {}

    def _load_file(self, file_path: str) -> List[Dict[str, Any]]:
        if file_path in self.source_cache:
            return self.source_cache[file_path]

        records = []
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    data = json.loads(line)
                    if "id" not in data:
                        data["id"] = f"line_{i}"
                    records.append(data)
                except: continue
        
        self.source_cache[file_path] = records
        return records

    def get_materials(self, file_path: str, used_ids: set[str], batch_size: int) -> List[Dict[str, Any]]:
        all_records = self._load_file(file_path)
        available = [r for r in all_records if str(r["id"]) not in used_ids]
        if not available: return []
        random.shuffle(available)
        return available[:batch_size]

class LLMEngine:
    def __init__(self, model_name: str, api_key: str, api_url: str, temperature: float = 0.1):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, base_url=api_url)
        self.temperature = temperature

    def call(self, system_content: str, user_content: str):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ],
            temperature=self.temperature,
            stream=False
        )
        return response.choices[0].message.content

    @retry(wait_fixed=3000, stop_max_attempt_number=3)
    def retry_call(self, system_content: str, user_content: str):
        return self.call(system_content, user_content)

class ContaminationFilter:
    def __init__(self, test_set_path: str, filter_field: str, n_gram: int = 20):
        self.n_gram = n_gram
        self.test_ngrams: set[str] = set()
        self.logger = logging.getLogger("ContaminationFilter")
        self._build_index(test_set_path, filter_field)

    def _clean_text(self, text: str) -> str:
        return re.sub(r'[^\w\u4e00-\u9fa5]', '', text) if text else ""

    def _build_index(self, test_set_path: str, filter_field: str):
        if not test_set_path or not os.path.exists(test_set_path):
            if test_set_path and test_set_path != "none.jsonl":
                self.logger.warning(f"测试集文件不存在或未配置，跳过污染索引构建: {test_set_path}")
            return
            
        self.logger.info(f"正在构建污染检测索引... 文件: {test_set_path}, 检测字段: '{filter_field}'")
        with open(test_set_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    text = data.get(filter_field, "")
                    cleaned = self._clean_text(text)
                    for i in range(len(cleaned) - self.n_gram + 1):
                        self.test_ngrams.add(cleaned[i : i + self.n_gram])
                except Exception as e: 
                    continue
        self.logger.info(f"污染检测索引构建完成，共生成 {len(self.test_ngrams)} 个 {self.n_gram}-grams。")

    def is_contaminated(self, material_text: str) -> bool:
        if not self.test_ngrams: 
            return False 
            
        cleaned = self._clean_text(material_text)
        if len(cleaned) < self.n_gram: return False
        
        for i in range(len(cleaned) - self.n_gram + 1):
            if cleaned[i : i + self.n_gram] in self.test_ngrams:
                return True
        return False

# --- 4. 主流水线 ---
class QASynthesizer:
    def __init__(self, templates: TemplateManager, data_loader: DataLoader, llm: LLMEngine, backup_dir: str = "backups"):
        self.templates = templates
        self.data_loader = data_loader
        self.llm = llm
        self.logger = logging.getLogger("QASynthesizer")
        
        self.backup_dir = backup_dir
        os.makedirs(self.backup_dir, exist_ok=True)
        # 修改：废弃记录升级为结构化的 jsonl 文件
        self.skipped_file = os.path.join(self.backup_dir, "skipped_records.jsonl")

    def run_pipeline(self, output_path: str):
        # 修改：将全局的 used_ids 改为按 format_id 隔离的字典结构
        used_ids_per_task = {fmt.format_id: set() for fmt in self.templates.formats}
        format_counts = {fmt.format_id: 0 for fmt in self.templates.formats}
        skipped_counts = {fmt.format_id: 0 for fmt in self.templates.formats}
        
        # 1. 从主输出文件恢复进度（成功的数据）
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        fid = data.get("format_id")
                        sid = str(data.get("source_id"))
                        if fid in used_ids_per_task and sid not in used_ids_per_task[fid]:
                            used_ids_per_task[fid].add(sid)
                            format_counts[fid] += 1
                    except: continue

        # 2. 从备份文件夹恢复进度（成功的数据，双重保险）
        for filename in os.listdir(self.backup_dir):
            if filename.endswith("_backup.jsonl"):
                backup_file_path = os.path.join(self.backup_dir, filename)
                with open(backup_file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            fid = data.get("format_id")
                            sid = str(data.get("source_id"))
                            if fid in used_ids_per_task and sid not in used_ids_per_task[fid]:
                                used_ids_per_task[fid].add(sid)
                                format_counts[fid] += 1
                        except: continue
        
        # 3. 从废弃记录文件中恢复进度（失败/跳过的数据）
        if os.path.exists(self.skipped_file):
            with open(self.skipped_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        fid = data.get("format_id")
                        sid = str(data.get("source_id"))
                        if fid in used_ids_per_task and sid not in used_ids_per_task[fid]:
                            used_ids_per_task[fid].add(sid)
                            skipped_counts[fid] += 1
                    except: continue
        
        print("🔄 进度恢复完成：")
        for fmt in self.templates.formats:
            fid = fmt.format_id
            print(f"   - 任务 [{fid}] | 已生成有效数据: {format_counts[fid]} 条 | 已跳过废弃记录: {skipped_counts[fid]} 条")

        # 使用追加模式打开主文件和废弃记录文件
        with open(output_path, 'a', encoding='utf-8') as f, \
             open(self.skipped_file, 'a', encoding='utf-8') as f_skip:
             
            for fmt in self.templates.formats:
                fid = fmt.format_id
                success_count = format_counts[fid]
                target_num = fmt.synthesis_num
                
                if success_count >= target_num:
                    print(f"✅ 任务 [{fid}] 已满足目标数量 ({target_num})，自动跳过。")
                    continue
                    
                print(f"▶️ 开始执行任务 [{fid}]，目标: {target_num}，当前有效进度: {success_count}...")
                
                current_filter = ContaminationFilter(
                    test_set_path=fmt.filter_path, 
                    filter_field=fmt.filter_field
                )

                backup_filepath = os.path.join(self.backup_dir, f"{fid}_backup.jsonl")
                with open(backup_filepath, 'a', encoding='utf-8') as bf:
                    
                    while success_count < target_num:
                        # 重点：此处只传入当前 format_id 对应的已用 ID 集合
                        materials = self.data_loader.get_materials(
                            file_path=fmt.source_path,
                            used_ids=used_ids_per_task[fid], 
                            batch_size=10
                        )
                        if not materials: 
                            print(f"⚠️ 警告：任务 [{fid}] 的可用数据源已耗尽！")
                            break
                            
                        for material in materials:
                            if success_count >= target_num: break
                            
                            source_id = str(material["id"])
                            material_text = material.get(fmt.content_field, "")
                            
                            def record_skipped(reason_str):
                                skip_record = {"format_id": fid, "source_id": source_id, "reason": reason_str}
                                f_skip.write(json.dumps(skip_record, ensure_ascii=False) + '\n')
                                f_skip.flush()
                                used_ids_per_task[fid].add(source_id)

                            # 异常分支 1: 文本为空
                            if not material_text:
                                record_skipped("empty_text")
                                continue

                            # 异常分支 2: 数据污染
                            if current_filter.is_contaminated(material_text):
                                msg = f"🛡️ 拦截警告: 检测到数据污染！Format: [{fid}], Source ID: [{source_id}] 的文本与测试集 [{fmt.filter_path}] 存在重合，已跳过。"
                                print(msg)
                                self.logger.warning(msg)
                                record_skipped("contaminated_data")
                                continue
                                
                            user_content = synthesize_user_template.format(
                                description=fmt.description, 
                                material=material_text
                            )
                            
                            try:
                                response = self.llm.retry_call(synthesize_system_prompt, user_content)
                                valid, parsed_qa = parse_synthesis_response(response)
                                
                                if valid:
                                    ans = parsed_qa.get("Ground-True Answer", "")
                                    if not isinstance(ans, list):
                                        ans = [ans]
                                        
                                    output_record = {
                                        "format_id": fid,
                                        "source_id": source_id,
                                        'Open-ended Verifiable Question': parsed_qa.get("Question", ""),
                                        'Ground-True Answer': ans
                                    }
                                    
                                    record_json = json.dumps(output_record, ensure_ascii=False) + '\n'
                                    
                                    # 写入主文件并强制落盘
                                    f.write(record_json)
                                    f.flush()
                                    
                                    # 写入备份文件并强制落盘
                                    bf.write(record_json)
                                    bf.flush()
                                    
                                    success_count += 1
                                    used_ids_per_task[fid].add(source_id)
                                    print(f"Progress [{fid}]: {success_count}/{target_num}")
                                else:
                                    # 异常分支 3: 模型判定不适用或泄露
                                    print(f"Validation failed or skipped for {source_id} (Mismatch, Leakage or Format error).")
                                    record_skipped("model_rejected_or_leakage")
                                    
                            except Exception as e:
                                print(f"Error processing {source_id}: {e}")

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("synthesis.log", encoding='utf-8'),
        ]
    )
    templates = TemplateManager(templates_config)
    data_loader = DataLoader()
    llm = LLMEngine(model_name=MODEL_NAME, api_key=API_KEY, api_url=API_URL)

    synthesizer = QASynthesizer(templates, data_loader, llm)

    print("开始生成测试用例...")

    synthesizer.run_pipeline(output_path=output_path)
    print(f"执行完成，请查看 {output_path} 及 backups/ 目录")

# 配置参数
API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
API_URL = os.getenv("OPENAI_API_URL", "YOUR_API_URL")
MODEL_NAME = "deepseek-v3"

if __name__ == "__main__":
    main()