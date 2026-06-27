#!/usr/bin/env bash
# Settle the "Gemma 128k batch1=1.81x, batch4=1.5x" recollection with a FRESH
# current-build measurement (post sliding-window auto-skip fix f43c622a4).
# fkv vs lrosa(fp8,cs64) @ 128k, batch=1 then batch=4. decode=64, n_fac=2048.
set -u
PY=/NHNHOME/jiwonsong/miniconda3/envs/vllm/bin/python
cd /NHNHOME/jiwonsong/vllm
export TMPDIR=/NHNHOME/jiwonsong/tmp LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}
export HF_HOME=/NHNHOME/jiwonsong/hf_cache HF_HUB_OFFLINE=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0
OUT=/NHNHOME/jiwonsong/tmp/latgemma_fresh; mkdir -p $OUT
SUM=$OUT/summary.log
M=google/gemma-4-26B-A4B-it
BASIS=/NHNHOME/jiwonsong/LRoSA-dev/bases/gemma_4_26b_a4b_it/pca_d1_cs64_kv_head_gemma_4_26b_a4b_it.pt
NFAC=2048; DECODE=64; PLEN=130816; CSH=64
run(){ # backend bsz gpu port
  local be=$1 bsz=$2 g=$3 port=$4 f=$OUT/${1}_128k_b${2}.txt
  local CR=$HOME/.cache/vllm_gf_${be}_${g}; rm -rf "$CR"
  local extra=""; [ "$be" = lrosa ] && extra="--basis $BASIS --cs_h $CSH --fp8_projk"
  CUDA_VISIBLE_DEVICES=$g VLLM_PORT=$port VLLM_CACHE_ROOT=$CR \
    TORCHINDUCTOR_CACHE_DIR=$TMPDIR/ind_gf_${be}_${g} TRITON_CACHE_DIR=$TMPDIR/tri_gf_${be}_${g} \
    $PY decode_latency_bench.py --backend $be --model $M --prefill_len $PLEN \
      --decode_len $DECODE --n_fac $NFAC --batch_size $bsz --gpu_mem 0.90 $extra \
      > $f 2>&1
}
echo "=== gemma fresh b1/b4 @128k start $(date +%H:%M:%S) ===" | tee -a $SUM
for bsz in 1 4; do
  echo "----- batch=$bsz $(date +%H:%M:%S) -----" | tee -a $SUM
  run fkv   $bsz 0 45110 &  P1=$!
  run lrosa $bsz 2 45120 &  P2=$!
  wait $P1 $P2
  for be in fkv lrosa; do
    f=$OUT/${be}_128k_b${bsz}.txt
    line=$(grep -hE "DECODE_MS_PER_TOK" $f 2>/dev/null | tail -1)
    err=$(grep -hiE "out of memory|CUDA error|RuntimeError|AssertionError|EADDRINUSE|Traceback" $f 2>/dev/null | tail -1)
    echo "RESULT b${bsz} 128k $be :: ${line:-FAILED: ${err:-see $f}}" | tee -a $SUM
  done
done
echo "=== gemma fresh done $(date +%H:%M:%S) ===" | tee -a $SUM