#!/usr/bin/env bash
# AIME25 / Qwen3-8B (thinking, YaRN on) / Quest (page min/max, token_budget=2048)
# on GPU 3. 8 attempts/problem -> pass@1. Resumable.
# Results: <PCA_REPO>/results/aime25/Qwen3-8B/quest/
source "$HOME/miniforge3/etc/profile.d/conda.sh" && conda activate vllm
set -u
cd /NHNHOME/jiwonsong/vllm
export TMPDIR=/NHNHOME/jiwonsong/tmp LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1
export HF_HUB_CACHE=/NHNHOME/jiwonsong/hf_cache/hub   # Qwen3-8B here; datasets in default cache
PY=$HOME/miniforge3/envs/vllm/bin/python

CUDA_VISIBLE_DEVICES=3 VLLM_CACHE_ROOT="$HOME/.cache/aime_quest" \
  "$PY" reasoning_vllm_eval.py --eval aime25 --mode quest --model Qwen/Qwen3-8B \
    --n_fac 1024 \
    --num_runs 8 --num_samples 0 --max_new_tokens 38912 \
    --max_num_seqs 8 --gpu_mem 0.85 --run_name quest_nf1024 \
  2>&1 | tee /NHNHOME/jiwonsong/tmp/aime_quest_nf1024.log
