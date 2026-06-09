#!/usr/bin/env bash
# run_glm_reason.sh <eval> <mode> <gpu> [num_samples=30]
# GLM-4.7-Flash (MLA) reasoning appendix: FKV vs LRoSA. mode in {fkv, lrosa_mla}.
#   fkv       -> TRITON_MLA (head-agnostic dense; GLM's 20 heads break trtllm FMHA)
#   lrosa_mla -> FLASHMLA_SPARSE + fp8_ds_mla (only path that fits 20 heads), cs_h=64,
#                n_fac=2048, LRoSAMLAIndexer scores the latent c_KV.
# CUDA_HOME/CPATH/nvcc needed for the flashinfer MoE JIT. Resumable (per run,idx).
ev=$1; md=$2; g=$3; ns=${4:-30}
cd /NHNHOME/jiwonsong/vllm
export CUDA_HOME=$HOME/miniforge3/envs/vllm PATH=$HOME/miniforge3/envs/vllm/bin:$PATH
export CPATH=$HOME/miniforge3/envs/vllm/targets/x86_64-linux/include:${CPATH:-}
export TMPDIR=/NHNHOME/jiwonsong/tmp HF_HUB_CACHE=/NHNHOME/jiwonsong/hf_cache/hub HF_HUB_OFFLINE=1
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$HOME/miniforge3/envs/vllm/targets/x86_64-linux/lib:${LIBRARY_PATH:-}
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0
PY=$HOME/miniforge3/envs/vllm/bin/python
BASIS=/NHNHOME/jiwonsong/LRoSA-dev/bases/glm_4_7_flash/pca_d1_cs64_kv_head_glm_4_7_flash.pt
case $md in
  fkv)       extra="--mla_backend TRITON_MLA"; rn=fkv;;
  lrosa_mla) extra="--mla_backend FLASHMLA_SPARSE --cs_h 64 --n_fac 2048 --basis $BASIS"; rn=lrosa_mla_cs64;;
  *) echo "bad mode $md"; exit 1;;
esac
CUDA_VISIBLE_DEVICES=$g VLLM_CACHE_ROOT="$HOME/.cache/glm_${ev}_${md}" \
  "$PY" reasoning_vllm_eval.py --eval "$ev" --mode "$md" --model zai-org/GLM-4.7-Flash $extra \
    --num_samples "$ns" --num_runs 1 --max_new_tokens 38912 --max_input_len 2048 \
    --eager --gpu_mem 0.85 --max_num_seqs 16 --run_name "$rn" \
  2>&1 | tee "/NHNHOME/jiwonsong/tmp/glm_${ev}_${md}.log"
