#!/usr/bin/env bash
# AIME25 / Qwen3-8B (thinking, YaRN on) / LRoSA (D1, cs_h=32, n_fac=2048,
# FP8 score by default) on GPU 2. 8 attempts/problem -> pass@1. Resumable.
# Results: <PCA_REPO>/results/aime25/Qwen3-8B/lrosa_cs32_fp8/
source "$HOME/miniforge3/etc/profile.d/conda.sh" && conda activate vllm
set -u
cd /NHNHOME/jiwonsong/vllm
export TMPDIR=/NHNHOME/jiwonsong/tmp LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1
export HF_HUB_CACHE=/NHNHOME/jiwonsong/hf_cache/hub   # Qwen3-8B here; datasets in default cache
PY=$HOME/miniforge3/envs/vllm/bin/python

# --fp8_projk is ON by default for lrosa (lossless vs bf16); add --no-fp8_projk for bf16.
CUDA_VISIBLE_DEVICES=2 VLLM_CACHE_ROOT="$HOME/.cache/aime_lrosa" \
  "$PY" reasoning_vllm_eval.py --eval aime25 --mode lrosa --model Qwen/Qwen3-8B \
    --cs_h 32 --n_fac 1024 \
    --num_runs 8 --num_samples 0 --max_new_tokens 38912 \
    --max_num_seqs 8 --gpu_mem 0.85 --run_name lrosa_cs32_fp8_nf1024 \
  2>&1 | tee /NHNHOME/jiwonsong/tmp/aime_lrosa_nf1024.log
