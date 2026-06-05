import os
import json
import time
import asyncio
import argparse
from typing import List, Dict, Optional

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def load_corpus(corpus_path: str) -> List[Dict]:
    print(f"[INFO] Loading corpus from {corpus_path}...")
    corpus = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                corpus.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    print(f"[INFO] Successfully loaded {len(corpus)} documents.")
    return corpus

class Config:
    def __init__(self, **kwargs):
        self.retrieval_model_path = kwargs.get("retrieval_model_path", "nvidia/llama-embed-nemotron-8b")
        self.corpus_path = kwargs.get("corpus_path", "")
        self.retrieval_topk = kwargs.get("topk", 5)
        self.batch_size = kwargs.get("batch_size", 4) 
        self.gpu_ids = kwargs.get("gpu_ids", 0)
        self.port = kwargs.get("port", 8215)
        self.max_concurrency = kwargs.get("max_concurrency", 4)


class UnifiedQueryItem(BaseModel):
    search_type: str = Field(alias="检索类型", default="法律检索")
    keywords: str = Field(alias="关键词")

class UnifiedQueryRequest(BaseModel):
    query: UnifiedQueryItem
    topk: Optional[int] = 5

# Llama-Nemotron 
class LlamaNemotronRetriever:
    def __init__(self, config: Config):
        self.config = config
        self.device = f"cuda:{config.gpu_ids}" if torch.cuda.is_available() else "cpu"
        self.topk = config.retrieval_topk
        self.batch_size = config.batch_size
        self.corpus_path = config.corpus_path
        
        print(f"[INFO] Llama-Nemotron locked on: {self.device}")
        self.embedder = SentenceTransformer(
            config.retrieval_model_path, 
            device=self.device, 
            trust_remote_code=True
        )
        
        self.corpus = load_corpus(self.corpus_path)
        self.embedding_file = self._get_embedding_filename(self.corpus_path)
        self.corpus_embeddings = self._load_or_compute_embeddings()

    def _get_embedding_filename(self, jsonl_filename: str) -> str:
        base_name = os.path.splitext(os.path.basename(jsonl_filename))[0]
        dir_name = os.path.dirname(jsonl_filename)
        return os.path.join(dir_name, f"{base_name}_llama_nemotron_embeddings.pt")

    def _load_or_compute_embeddings(self) -> torch.Tensor:
        if os.path.exists(self.embedding_file):
            print(f"[INFO] Loading existing Llama embeddings from {self.embedding_file}")
            return torch.load(self.embedding_file, map_location=self.device)
        print(f"[WARNING] Local embeddings not found. Computing new Llama embeddings...")
        return self._compute_and_save_embeddings()

    def _compute_and_save_embeddings(self) -> torch.Tensor:
        corpus_texts = [doc.get('contents') or doc.get('text') or doc.get('content') or str(doc) for doc in self.corpus]
        print(f"[INFO] Encoding {len(corpus_texts)} documents with Llama-Nemotron...")
        corpus_embeddings = self.embedder.encode(
            corpus_texts, 
            show_progress_bar=True, 
            normalize_embeddings=True, 
            batch_size=self.batch_size,
            convert_to_tensor=True
        )
        torch.save(corpus_embeddings, self.embedding_file)
        return corpus_embeddings

    def search_sync(self, query: str, num: int) -> tuple:
        if not query.strip(): return [], []
        query_embedding = self.embedder.encode(
            query, 
            normalize_embeddings=True, 
            convert_to_tensor=True
        )
        hits = semantic_search(query_embedding, self.corpus_embeddings, top_k=num)[0]
        results, scores = [], []
        for hit in hits:
            results.append(self.corpus[hit['corpus_id']])
            scores.append(hit['score'])
        return results, scores


parser = argparse.ArgumentParser()
parser.add_argument("--corpus_path", type=str, required=True)
parser.add_argument("--retriever_model", type=str, required=True)
parser.add_argument("--topk", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--port", type=int, default=8215)
parser.add_argument("--gpu_ids", type=int, default=0)
parser.add_argument("--max_concurrency", type=int, default=4)
args, _ = parser.parse_known_args()

global_config = Config(
    corpus_path=args.corpus_path,
    retrieval_model_path=args.retriever_model,
    topk=args.topk, batch_size=args.batch_size,
    port=args.port, gpu_ids=args.gpu_ids, max_concurrency=args.max_concurrency
)

app = FastAPI()
law_retriever = None
gpu_semaphore = None

def get_gpu_semaphore():
    global gpu_semaphore
    if gpu_semaphore is None: gpu_semaphore = asyncio.Semaphore(global_config.max_concurrency)
    return gpu_semaphore

@app.on_event("startup")
async def startup_event():
    global law_retriever
    print(f"[INFO] Worker starting, loading Llama-Nemotron-8B...")
    law_retriever = LlamaNemotronRetriever(global_config)

@app.post("/retrieve")
async def retrieve_endpoint(request: UnifiedQueryRequest):
    req_type = request.query.search_type
    keywords = request.query.keywords
    req_topk = request.topk if request.topk else global_config.retrieval_topk
    
    if req_type != "法律检索": return {"error": f"unsupported:{req_type}"}
        
    async with get_gpu_semaphore():
        results, scores = await asyncio.to_thread(law_retriever.search_sync, keywords, req_topk)
    
    processed_docs = []
    for i, doc in enumerate(results):
        standard_doc = doc.copy()
        standard_doc['content'] = doc.get('content') or doc.get('contents') or doc.get('text') or ""
        processed_docs.append({"document": standard_doc, "score": round(scores[i], 4)})
        
    return {"检索类型": "法律检索", "result": processed_docs}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=global_config.port, workers=1)