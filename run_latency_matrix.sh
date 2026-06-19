#!/usr/bin/env bash
# batch=1 decode-latency matrix: {fkv,fasa,lrosa,quest} x {16k,32k,64k,128k}
# on Qwen3-8B (n_fac=2048, YaRN). 4 backends run in parallel (one per GPU)
# per context round. Results -> /NHNHOME/jiwonsong/tmp/latbench/.
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate vllm
cd /NHNHOME/jiwonsong/vllm
export TMPDIR=/NHNHOME/jiwonsong/tmp LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}
export HF_HOME=/NHNHOME/jiwonsong/hf_cache
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0
OUT=/NHNHOME/jiwonsong/tmp/latbench
MODEL=Qwen/Qwen3-8B
NFAC=2048
DECODE=128
BSZ=${BSZ:-1}
declare -A CTX=( [16k]=16384 [32k]=32768 [64k]=65536 [128k]=130816 )
BACKENDS=(${BES:-fkv fasa lrosa quest})

echo "=== latency matrix start (bsz=$BSZ) $(date +%H:%M:%S) ==="
for label in ${CTXS:-16k 32k 64k 128k}; do
  plen=${CTX[$label]}
  echo "----- context $label ($plen) $(date +%H:%M:%S) -----"
  gpu=0
  for be in "${BACKENDS[@]}"; do
    f=$OUT/${be}_${label}_b${BSZ}.txt
    # Per-backend compile-cache dir: concurrent procs sharing one
    # torch_compile_cache race and can corrupt a kernel -> illegal access.
    CUDA_VISIBLE_DEVICES=$gpu VLLM_CACHE_ROOT=$HOME/.cache/vllm_${be} \
      python decode_latency_bench.py \
      --backend $be --model $MODEL --prefill_len $plen --decode_len $DECODE \
      --n_fac $NFAC --n_tip 16 --batch_size $BSZ --gpu_mem 0.92 \
      > $f 2>&1 &
    gpu=$((gpu+1))
  done
  wait
  for be in "${BACKENDS[@]}"; do
    f=$OUT/${be}_${label}_b${BSZ}.txt
    line=$(grep -hE "DECODE_MS_PER_TOK" $f 2>/dev/null | tail -1)
    err=$(grep -hiE "out of memory|OutOfMemory|CUDA error|Error|Traceback" $f 2>/dev/null | tail -1)
    echo "RESULT b${BSZ} $label $be :: ${line:-FAILED: ${err}}"
  done
done
echo "=== latency matrix done $(date +%H:%M:%S) ==="
