#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Max-batch + decode-throughput sweep for one (backend, context) on one GPU.
#
# Doubles batch_size (1,2,4,8,...) running decode_latency_bench.py at each,
# until the run OOMs (or hits MAX_BSZ). For every batch it records
# DECODE_MS_PER_TOK / PER_STREAM_TOK_S / AGG_TOK_S (= aggregate tokens/s, the
# throughput). Reports the largest batch that fit (max batch) and the batch
# with the highest AGG_TOK_S (peak throughput — not always the max batch, since
# decode is HBM-bound and throughput can plateau).
#
# Usage:
#   GPU=0 BACKEND=lrosa CTX=65536 MODEL=Qwen/Qwen3-8B NFAC=2048 \
#     bash sweep_max_batch_throughput.sh
#
# Env knobs (all optional except BACKEND, CTX):
#   GPU=0            CUDA device index
#   BACKEND         fkv | lrosa | fasa | quest | lrosa_mla   (required)
#   CTX=65536       prefill_len (context length); required
#   MODEL           HF id (default Qwen/Qwen3-8B)
#   NFAC=2048       token budget (n_fac)  | CS_H=32 N_TIP=16
#   DECODE=128      decode steps to time
#   GPU_MEM=0.9     gpu_memory_utilization
#   BSZ_START=1  MAX_BSZ=512   batch sweep bounds (doubling)
#   BASIS=...       basis .pt (REQUIRED for lrosa/fasa/lrosa_mla if not at the
#                   default PCA_REPO/bases/<tag>/ path — e.g. GLM lives in
#                   bases_la_full/). PCA_REPO overrides the bases root.
#   MLA_KV_DTYPE=fp8_ds_mla   MLA_BACKEND=FLASHMLA_SPARSE  (MLA models)
#   CHUNKED_PREFILL=1         add --chunked_prefill (MLA ctx>=32k: avoids the
#                             prefill-workspace OOM; lets the real decode batch fit)
#   MLA_FKV_FP8=1             fkv on an MLA model: use the same fp8 latent KV
#   PY=...          python (default the vllm conda env)
#   OUT=...         results dir (default /NHNHOME/jiwonsong/tmp/tput_sweep)
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$0")"

: "${BACKEND:?set BACKEND=fkv|lrosa|fasa|quest|lrosa_mla}"
: "${CTX:?set CTX=<prefill_len, e.g. 65536>}"
GPU=${GPU:-0}
MODEL=${MODEL:-Qwen/Qwen3-8B}
NFAC=${NFAC:-2048}; CS_H=${CS_H:-32}; N_TIP=${N_TIP:-16}
DECODE=${DECODE:-128}
GPU_MEM=${GPU_MEM:-0.9}
BSZ_START=${BSZ_START:-1}; MAX_BSZ=${MAX_BSZ:-512}
PY=${PY:-/home/snu_open/miniforge3/envs/vllm/bin/python}
OUT=${OUT:-/NHNHOME/jiwonsong/tmp/tput_sweep}
mkdir -p "$OUT"

# Env the GLM/MLA + flashinfer-MoE JIT path needs (harmless for GQA models).
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} PATH="$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include${CPATH:+:$CPATH}"
# Per-(backend,ctx) compile cache: concurrent/!shared caches race & corrupt kernels.
tag="${MODEL##*/}_${BACKEND}_c${CTX}"
export VLLM_CACHE_ROOT=/NHNHOME/jiwonsong/tmp/vcr_tput_${tag}
export TRITON_CACHE_DIR=/NHNHOME/jiwonsong/tmp/tcr_tput_${tag}
mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR"

extra=()
[ -n "${BASIS:-}" ] && extra+=(--basis "$BASIS")
[ "$BACKEND" = "lrosa" ] || [ "$BACKEND" = "lrosa_mla" ] && extra+=(--cs_h "$CS_H")
[ "$BACKEND" = "fasa" ] && extra+=(--n_tip "$N_TIP")
[ -n "${MLA_KV_DTYPE:-}" ] && extra+=(--mla_kv_dtype "$MLA_KV_DTYPE")
[ -n "${MLA_BACKEND:-}" ] && extra+=(--mla_backend "$MLA_BACKEND")
[ "${CHUNKED_PREFILL:-0}" = "1" ] && extra+=(--chunked_prefill)
[ "${MLA_FKV_FP8:-0}" = "1" ] && extra+=(--mla_fkv_fp8)

summary="$OUT/SUMMARY_${tag}.txt"; : > "$summary"
echo "# sweep $MODEL backend=$BACKEND ctx=$CTX nfac=$NFAC decode=$DECODE gpu_mem=$GPU_MEM gpu=$GPU" | tee "$summary"
echo "# bsz   decode_ms/tok   per_stream_tok/s   AGG_tok/s(throughput)   status" | tee -a "$summary"

best_bsz=0; best_tput=0; max_ok=0; bsz=$BSZ_START
while [ "$bsz" -le "$MAX_BSZ" ]; do
  log="$OUT/${tag}_b${bsz}.log"
  CUDA_VISIBLE_DEVICES=$GPU "$PY" decode_latency_bench.py \
    --backend "$BACKEND" --model "$MODEL" --prefill_len "$CTX" --decode_len "$DECODE" \
    --n_fac "$NFAC" --batch_size "$bsz" --gpu_mem "$GPU_MEM" "${extra[@]}" \
    > "$log" 2>&1
  rc=$?
  line=$(grep -hE "DECODE_MS_PER_TOK" "$log" 2>/dev/null | tail -1)
  if [ -n "$line" ]; then
    ms=$(echo "$line"   | grep -oE "DECODE_MS_PER_TOK=[0-9.]+" | cut -d= -f2)
    pst=$(echo "$line"  | grep -oE "PER_STREAM_TOK_S=[0-9.]+" | cut -d= -f2)
    agg=$(echo "$line"  | grep -oE "AGG_TOK_S=[0-9.]+"        | cut -d= -f2)
    printf "%-6s %-15s %-18s %-22s OK\n" "$bsz" "$ms" "$pst" "$agg" | tee -a "$summary"
    max_ok=$bsz
    awk "BEGIN{exit !($agg > $best_tput)}" && { best_tput=$agg; best_bsz=$bsz; }
  else
    if grep -qiE "out of memory|OutOfMemory|CUDA error|CUDA out of memory|HIP out of memory" "$log"; then
      printf "%-6s %-15s %-18s %-22s OOM (stop)\n" "$bsz" "-" "-" "-" | tee -a "$summary"
      break
    else
      printf "%-6s %-15s %-18s %-22s FAILED(non-OOM, see %s)\n" "$bsz" "-" "-" "-" "$log" | tee -a "$summary"
      break
    fi
  fi
  bsz=$((bsz * 2))
done

echo "----" | tee -a "$summary"
echo "MAX_BATCH=$max_ok  PEAK_THROUGHPUT_TOK_S=$best_tput @ bsz=$best_bsz" | tee -a "$summary"
echo "(full per-bsz logs in $OUT/${tag}_b*.log; summary -> $summary)"
