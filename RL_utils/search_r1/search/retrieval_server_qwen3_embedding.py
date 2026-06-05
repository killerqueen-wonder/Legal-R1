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

# ==========================================
# Core Components: Data Loading and Global Configuration
# ==========================================

def load_corpus(corpus_path: str) -> List[Dict]:
    print(f"[INFO] Loading corpus from {corpus_path}...")
    corpus = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                corpus.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    print(f"[INFO] Successfully loaded {len(corpus)} documents.")
    return corpus

class Config:
    def __init__(self, **kwargs):
        self.retrieval_model_path = kwargs.get("retrieval_model_path", "Qwen/Qwen3-Embedding-8B")
        self.corpus_path = kwargs.get("corpus_path", "")
        self.retrieval_topk = kwargs.get("topk", 5)
        self.batch_size = kwargs.get("batch_size", 128)
        self.gpu_ids = kwargs.get("gpu_ids", 0)
        self.port = kwargs.get("port", 8006)
        self.max_concurrency = kwargs.get("max_concurrency", 4)

# ==========================================
# FastAPI Input/Output Validation Models
# ==========================================

class UnifiedQueryItem(BaseModel):
    search_type: str = Field(alias="检索类型", default="法律检索")
    keywords: str = Field(alias="关键词")

class UnifiedQueryRequest(BaseModel):
    query: UnifiedQueryItem
    topk: Optional[int] = 5

# ==========================================
# Core Business Class: Qwen3-Embedding Retriever
# ==========================================

class Qwen3Text2vecRetriever:
    def __init__(self, config: Config):
        self.config = config
        self.device = f"cuda:{config.gpu_ids}" if torch.cuda.is_available() else "cpu"
        self.topk = config.retrieval_topk
        self.batch_size = config.batch_size
        self.corpus_path = config.corpus_path
        
        print(f"[INFO] Qwen3-Embedding strictly locked on: {self.device}")
        self.embedder = SentenceTransformer(config.retrieval_model_path, device=self.device)
        
        self.corpus = load_corpus(self.corpus_path)
        self.embedding_file = self._get_embedding_filename(self.corpus_path)
        
        # Check on startup: automatically compute if no vector cache exists
        self.corpus_embeddings = self._load_or_compute_embeddings()

    def _get_embedding_filename(self, jsonl_filename: str) -> str:
        """Generate an exclusive .pt suffix for Qwen3 to avoid vector conflicts with old models"""
        base_name = os.path.splitext(os.path.basename(jsonl_filename))[0]
        dir_name = os.path.dirname(jsonl_filename)
        return os.path.join(dir_name, f"{base_name}_qwen3_embeddings.pt")

    def _load_or_compute_embeddings(self) -> torch.Tensor:
        if os.path.exists(self.embedding_file):
            print(f"[INFO] Loading existing Qwen3 embeddings from {self.embedding_file}")
            return torch.load(self.embedding_file, map_location=self.device)
        
        print(f"[WARNING] Local embeddings not found. Computing new Qwen3 embeddings in real-time...")
        return self._compute_and_save_embeddings()

    def _compute_and_save_embeddings(self) -> torch.Tensor:
        # Extract knowledge base text content to maintain integrity
        corpus_texts = [doc.get('contents') or doc.get('text') or doc.get('content') or str(doc) for doc in self.corpus]
        
        # [Core Note]: The official documentation explicitly requires NOT adding a prompt when building the knowledge base
        print(f"[INFO] Encoding {len(corpus_texts)} documents (this might take a while)...")
        corpus_embeddings = self.embedder.encode(
            corpus_texts, 
            show_progress_bar=True, 
            normalize_embeddings=True, 
            batch_size=self.batch_size,
            convert_to_tensor=True
        )
        torch.save(corpus_embeddings, self.embedding_file)
        print(f"[INFO] Embeddings computed and saved to {self.embedding_file}")
        return corpus_embeddings

    def search_sync(self, query: str, num: int) -> tuple:
        """Synchronous underlying retrieval function"""
        if not query.strip():
            return [], []
        
        # [Core Note]: The official documentation explicitly requires forcefully adding prompt_name="query" when retrieving a Query
        query_embedding = self.embedder.encode(
            query, 
            prompt_name="query", 
            normalize_embeddings=True, 
            convert_to_tensor=True
        )
        
        hits = semantic_search(query_embedding, self.corpus_embeddings, top_k=num)[0]
        
        results, scores = [], []
        for hit in hits:
            doc = self.corpus[hit['corpus_id']]
            results.append(doc)
            scores.append(hit['score'])
            
        return results, scores


# ==========================================
# Global Variables and FastAPI Application Initialization
# ==========================================

# Reserved for argparse
parser = argparse.ArgumentParser(description="Qwen3-Embedding RAG Retriever.")
parser.add_argument("--corpus_path", type=str, required=True, help="Knowledge base corpus path (.jsonl)")
parser.add_argument("--retriever_model", type=str, default="Qwen/Qwen3-Embedding-8B", help="Qwen3 model path")
parser.add_argument("--topk", type=int, default=5, help="Default number of retrieval results")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size when building the database")
parser.add_argument("--port", type=int, default=8006, help="API port number")
parser.add_argument("--gpu_ids", type=int, default=0, help="Bound GPU ID")
parser.add_argument("--max_concurrency", type=int, default=4, help="Maximum number of concurrent request locks")

args, unknown = parser.parse_known_args()

global_config = Config(
    corpus_path=args.corpus_path,
    retrieval_model_path=args.retriever_model,  
    topk=args.topk,
    batch_size=args.batch_size,
    port=args.port,
    gpu_ids=args.gpu_ids,
    max_concurrency=args.max_concurrency
)

app = FastAPI()
law_retriever = None
gpu_semaphore = None

def get_gpu_semaphore():
    """Lazy load GPU semaphore"""
    global gpu_semaphore
    if gpu_semaphore is None:
        gpu_semaphore = asyncio.Semaphore(global_config.max_concurrency)
    return gpu_semaphore

@app.on_event("startup")
async def startup_event():
    global law_retriever
    pid = os.getpid()
    print(f"[INFO] Worker {pid} is starting and loading Qwen3-Embedding model into VRAM...")
    
    # Initialization will automatically validate and compute vector files
    law_retriever = Qwen3Text2vecRetriever(global_config)
    
    print(f"[INFO] Worker {pid} model loaded, service ready!")


# ==========================================
# API Route Definitions
# ==========================================

@app.post("/retrieve")
async def unified_retrieve_endpoint(request: UnifiedQueryRequest):
    print(f"[INFO] Received new request, extracted keywords: '{request.query.keywords}'")
    start_time = time.time()
    
    req_type = request.query.search_type
    keywords = request.query.keywords
    req_topk = request.topk if request.topk else global_config.retrieval_topk
    
    if req_type != "法律检索":
        return {"error": f"only support '法律检索',unsupport:{req_type}"}
        
    # Acquire concurrency lock to prevent out-of-memory from excessive simultaneous requests
    async with get_gpu_semaphore():
        # Offload tensor computation to background thread to prevent blocking FastAPI's async main event loop
        results, scores = await asyncio.to_thread(
            law_retriever.search_sync, 
            keywords, 
            req_topk
        )
    
    # Standardize output format
    processed_docs = []
    for i, doc in enumerate(results):
        standard_doc = doc.copy()
        # Uniformly pad the content field to ensure format compatibility
        standard_doc['content'] = doc.get('content') or doc.get('contents') or doc.get('text') or ""
        processed_docs.append({
            "document": standard_doc, 
            "score": round(scores[i], 4)  # Keep four decimal places
        })
        
    end_time = time.time()
    print(f"[DEBUG] Retrieval completed, time taken: {end_time - start_time:.4f}s")
    
    return {
        "检索类型": "法律检索",
        "result": processed_docs
    }

if __name__ == "__main__":
    print("[INFO] Starting Qwen3-Embedding Server...")
    uvicorn.run(app, host="0.0.0.0", port=global_config.port, workers=1)