#!/usr/bin/env bash
# AIME25 / Qwen3-8B (thinking, YaRN on) / FKV (dense) on GPU 1.
# 8 attempts per problem -> pass@1 (avg). Resumable: re-running continues
# from the last completed (problem, attempt) in predictions.jsonl.
# Results: <PCA_REPO>/results/aime25/Qwen3-8B/fkv/{summary.json,predictions.jsonl}
source "$HOME/miniforge3/etc/profile.d/conda.sh" && conda activate vllm
set -u
cd /NHNHOME/jiwonsong/vllm
export TMPDIR=/NHNHOME/jiwonsong/tmp LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1
export HF_HUB_CACHE=/NHNHOME/jiwonsong/hf_cache/hub   # Qwen3-8B here; datasets in default cache
PY=$HOME/miniforge3/envs/vllm/bin/python

CUDA_VISIBLE_DEVICES=1 VLLM_CACHE_ROOT="$HOME/.cache/aime_fkv" \
  "$PY" reasoning_vllm_eval.py --eval aime25 --mode fkv --model Qwen/Qwen3-8B \
    --num_runs 8 --num_samples 0 --max_new_tokens 38912 \
    --max_num_seqs 8 --gpu_mem 0.85 \
  2>&1 | tee /NHNHOME/jiwonsong/tmp/aime_fkv.log
