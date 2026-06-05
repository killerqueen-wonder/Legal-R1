#!/bin/bash
# Suitable for 4x H20 (141GB VRAM) Extreme Compute Edition - [3+1 Physically Isolated Architecture]
# GPU 3: Dedicated for Reward + Filter + Retriever (Infrastructure GPU)
# GPU 0, 1, 2: Dedicated for FSDP Training + Rollout (Pure Training GPUs)
# Description: ppo_mini_batch_size=6, number of generated responses n=4, number of GPUs = 3,
# Global single computation load: 6 (num prompts) * 4 (num responses) = 24
# Allocated per GPU (Normalized): 24 / 3 GPUs = 8

# Dynamically get the project root directory (Legal-R1)
export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Use relative path for data
export DATA_DIR="./data/train_data/RL"


export BASE_MODEL="/path/to/your/model"
export REWARD_MODEL="/path/to/your/model"
export FILTER_MODEL="/path/to/your/model"
export CONDA_SH="/path/to/your/conda.sh"

export EXPERIMENT_NAME=Legal-R1
export WAND_PROJECT='Legal-R1'
export WANDB_API_KEY=''

# Port Configuration
RETRIEVER_PORT=8005
VLLM_PORT=8006
REWARD_PORT=9000 

# Timeout and Distributed Environment Settings
PORT_TIMEOUT=1800 
export NCCL_TIMEOUT=3600
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=3600
export NCCL_SHM_DISABLE=1
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1

info() { echo -e "\033[32m[INFO]\033[0m $1"; }
error() { echo -e "\033[31m[ERROR]\033[0m $1"; }

wait_for_port() {
    local port=$1
    local timeout=$2
    local session_name=$3
    local start=$(date +%s)
    info "Waiting for port $port to be ready (timeout ${timeout}s)..."
    
    while true; do
        if (echo > /dev/tcp/127.0.0.1/$port) >/dev/null 2>&1; then
            info "Port $port is ready"
            return 0
        fi
        local now=$(date +%s)
        if [ $((now - start)) -ge $timeout ]; then
            error "Port $port not ready within ${timeout}s. Check logs: tmux attach -t $session_name"
            return 1
        fi
        sleep 5
    done
}

kill_session() {
    if tmux has-session -t "$1" 2>/dev/null; then
        info "Cleaning up historical session: $1"
        tmux kill-session -t "$1"
        sleep 1
    fi
}

tmux_send_commands() {
    local session=$1
    shift
    tmux send-keys -t "$session" "source $CONDA_SH" C-m
    for cmd in "$@"; do
        tmux send-keys -t "$session" "$cmd" C-m
    done
}

if ! command -v tmux &> /dev/null; then
    info "tmux not detected, attempting to install..."
    apt update && apt install -y tmux
fi

# ==================== 1. Start underlying dependency services (all centralized on GPU 3) ====================
info "Starting underlying services (deployed on GPU 3 infrastructure GPU)..."

# Component A: Reward LLM (GPU 3)
kill_session "reward_llm"
tmux new-session -d -s reward_llm -n reward
tmux_send_commands "reward_llm" \
    "export CUDA_VISIBLE_DEVICES=3" \
    "conda activate vllm_server" \
    "export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH" \
    "python -m vllm.entrypoints.openai.api_server \
        --model $REWARD_MODEL \
        --served-model-name qwen3-8b-reward \
        --host 0.0.0.0 --port $REWARD_PORT \
        --enable-prefix-caching \
        --enable-chunked-prefill \
        --max-num-seqs 64 \
        --max-model-len 33000 \
        --gpu-memory-utilization 0.5 \
        --dtype bfloat16 \
        --trust-remote-code"

# Component B: RAG Retrieval Filter Service (GPU 3)
kill_session "retriever_filter8005"
tmux new-session -d -s retriever_filter8005 -n retriever
tmux_send_commands "retriever_filter8005" \
    "conda activate retriever_filter" \
    "export TRANSFORMERS_CACHE=/path/to/your/cache_dir" \
    "export HF_HUB_CACHE=/path/to/your/cache_dir" \
    "export TRANSFORMERS_OFFLINE=1" \
    "export HF_HUB_OFFLINE=1" \
    "cd $PROJECT_ROOT" \
    "python search_r1/search/async_retrieval_server.py \
        --port $RETRIEVER_PORT \
        --corpus_path './data/RAG/legal_corpus.jsonl' \
        --case_corpus_path './data/RAG/Criminal Precedent Corpus_psi_subset.jsonl' \
        --retriever_name hybrid_filter \
        --dictionary_path './data/RAG/dictionary.txt' \
        --search_depth 5 --bm25_weight 15 --bm25_weight_factor 2 --bm25_k1 0.15 --bm25_b 0.35 --topk 8 \
        --retriever_model 'shibing624/text2vec-base-chinese-paraphrase' \
        --filter_model $FILTER_MODEL \
        --gpu_ids 3 --gpu_memory_limit_per_gpu 5"

sleep 15

# Component C: Filter/Rerank Service (GPU 3)
kill_session "vllm"
tmux new-session -d -s vllm -n vllm
tmux_send_commands "vllm" \
    "export CUDA_VISIBLE_DEVICES=3" \
    "conda activate vllm_server" \
    "export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH" \
    "python -m vllm.entrypoints.openai.api_server \
        --model $FILTER_MODEL \
        --served-model-name Qwen3-8B \
        --port $VLLM_PORT \
        --enable-chunked-prefill \
        --max-model-len 12000 \
        --gpu-memory-utilization 0.40 \
        --max-num-seqs 64 \
        --dtype bfloat16 \
        --trust-remote-code"

# ==================== 2. Health Check ====================
info "Waiting for all dependency services to be ready..."

wait_for_port $REWARD_PORT $PORT_TIMEOUT "reward_llm" || exit 1
wait_for_port $VLLM_PORT $PORT_TIMEOUT "vllm" || exit 1
wait_for_port $RETRIEVER_PORT $PORT_TIMEOUT "retriever_filter8005" || exit 1

# ==================== 3. Start Main Training Task (GPU 0, 1, 2) ====================
info "======================================================"
info "Background components (GPU 3) started, officially starting main training (GPU 0,1,2)! 🚀"
info "======================================================"

export CUDA_VISIBLE_DEVICES=0,1,2

# Run from the project root to ensure all relative paths map correctly
cd "$PROJECT_ROOT"

# Note: data.train_files updated to match 'train_subset.parquet' from your file structure
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train_subset.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=9 \
    data.val_batch_size=9 \
    data.max_prompt_length=25000 \
    data.max_response_length=1200 \
    +data.max_start_length=4000 \
    +data.max_obs_length=2000 \
    +data.shuffle_train_dataloader=true \
    data.filter_overlong_prompts=true \
    data.truncation='middle' \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=9 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=6 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.08 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=6 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.60 \
    actor_rollout_ref.rollout.enable_prefix_caching=true \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=6 \
    actor_rollout_ref.ref.fsdp_config.param_offload=false \
    actor_rollout_ref.rollout.temperature=0.5 \
    actor_rollout_ref.rollout.max_num_batched_tokens=60000 \
    algorithm.use_kl_in_reward=false \
    +algorithm.no_think_rl=false \
    +trainer.use_critic=false \
    +trainer.do_search=true \
    trainer.critic_warmup=0 \
    trainer.logger='["wandb"]' \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.val_before_train=false \
    trainer.n_gpus_per_node=3 \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.test_freq=30 \
    trainer.total_epochs=2 \
    trainer.total_training_steps=1801 \
    trainer.resume_mode=auto \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=./checkpoints/$EXPERIMENT_NAME \
    +max_turns=9 \
    ray_kwargs.ray_init.num_cpus=16 \
    +retriever.url="http://127.0.0.1:8005/retrieve" \
    +retriever.topk=8 \
    2>&1 | tee $EXPERIMENT_NAME.log