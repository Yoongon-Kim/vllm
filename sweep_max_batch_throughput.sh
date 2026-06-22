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
[ "${FP8_PROJK:-0}" = "1" ] && extra+=(--fp8_projk)   # LRoSA proj_K in a SEPARATE fp8 cache

summary="$OUT/SUMMARY_${tag}.txt"; : > "$summary"
echo "# sweep $MODEL backend=$BACKEND ctx=$CTX nfac=$NFAC decode=$DECODE gpu_mem=$GPU_MEM gpu=$GPU" | tee "$summary"
echo "# bsz   decode_ms/tok   per_stream_tok/s   AGG_tok/s(throughput)   status" | tee -a "$summary"

# Run one batch_size and emit its summary row. $1 = bsz.
run_bsz() {
  local bsz=$1 log="$OUT/${tag}_b${1}.log"
  CUDA_VISIBLE_DEVICES=$GPU "$PY" decode_latency_bench.py \
    --backend "$BACKEND" --model "$MODEL" --prefill_len "$CTX" --decode_len "$DECODE" \
    --n_fac "$NFAC" --batch_size "$bsz" --gpu_mem "$GPU_MEM" "${extra[@]}" \
    > "$log" 2>&1
  local line; line=$(grep -hE "DECODE_MS_PER_TOK" "$log" 2>/dev/null | tail -1)
  if [ -n "$line" ]; then
    local ms pst agg
    ms=$(echo "$line"  | grep -oE "DECODE_MS_PER_TOK=[0-9.]+" | cut -d= -f2)
    pst=$(echo "$line" | grep -oE "PER_STREAM_TOK_S=[0-9.]+" | cut -d= -f2)
    agg=$(echo "$line" | grep -oE "AGG_TOK_S=[0-9.]+"        | cut -d= -f2)
    printf "%-6s %-15s %-18s %-22s OK\n" "$bsz" "$ms" "$pst" "$agg" | tee -a "$summary"
    awk "BEGIN{exit !($agg > $best_tput)}" && { best_tput=$agg; best_bsz=$bsz; }
    LAST_KVTOK=$(grep -oE "GPU KV cache size: [0-9,]+ tokens" "$log" 2>/dev/null | grep -oE "[0-9,]+" | tr -d ',' | head -1)
    return 0
  fi
  if grep -qiE "out of memory|OutOfMemory|CUDA error|HIP out of memory" "$log"; then
    printf "%-6s %-15s %-18s %-22s OOM\n" "$bsz" "-" "-" "-" | tee -a "$summary"
  else
    printf "%-6s %-15s %-18s %-22s FAILED(non-OOM, see %s)\n" "$bsz" "-" "-" "-" "$log" | tee -a "$summary"
  fi
  return 1
}

# Concurrent max batch is the KV-pool capacity, NOT an OOM sweep: vLLM
# QUEUES/PREEMPTS requests that exceed the KV pool, so generate() still
# completes at batch >> capacity (no OOM) — an OOM-doubling sweep therefore
# measures the preemption/activation ceiling, not what fits CONCURRENTLY
# (badly over-stated at long ctx, where capacity is single digits). Instead:
# probe with batch=1, read the engine's "GPU KV cache size: N tokens", and
# CONCURRENT_MAX_BATCH = floor(N / (ctx + decode)). Throughput is then measured
# only at batches that actually fit (<= concurrent max); beyond that the timing
# reflects preemption, not concurrency.
best_bsz=0; best_tput=0; LAST_KVTOK=""
run_bsz "$BSZ_START" || true
kvtok=${LAST_KVTOK:-0}
per_req=$(( CTX + DECODE ))
conc_max=0
[ "${kvtok:-0}" -gt 0 ] && conc_max=$(( kvtok / per_req ))
echo "# GPU_KV_CACHE_TOKENS=$kvtok  per_request=$per_req  -> CONCURRENT_MAX_BATCH=$conc_max" | tee -a "$summary"

# Throughput at fitting batches: powers of 2 in (BSZ_START, min(conc_max,MAX_BSZ)]
# plus conc_max itself (the true capacity point).
cap=$conc_max; [ "$MAX_BSZ" -lt "$cap" ] && cap=$MAX_BSZ
b=$(( BSZ_START * 2 ))
while [ "$b" -lt "$cap" ]; do run_bsz "$b" || break; b=$(( b * 2 )); done
[ "$cap" -gt "$BSZ_START" ] && run_bsz "$cap" || true

echo "----" | tee -a "$summary"
echo "CONCURRENT_MAX_BATCH=$conc_max  (KV-fit, preemption-independent)" | tee -a "$summary"
echo "PEAK_THROUGHPUT_TOK_S=$best_tput @ bsz=$best_bsz  (within the concurrent regime)" | tee -a "$summary"

# --- Memory-fair concurrent max batch --------------------------------------
# CONCURRENT_MAX_BATCH above comes from the KV pool, which vLLM sizes from the
# per-token KV SLOT only. Methods with an UNCOUNTED separate side-buffer (Quest
# page min/max; LRoSA fp8 proj_K contig cache; Seer gate cache) consume extra
# HBM NOT in the slot -> it eats gpu_mem headroom, so their CONCURRENT_MAX_BATCH
# is OVER-stated vs FKV (and vs in-slot bf16 LRoSA proj_K, which IS counted).
# Report the iso-memory-fair value = CONCURRENT_MAX_BATCH * slot/(slot+side),
# i.e. what fits once the side-buffer is charged like the slot.
HEAD_SIZE=${HEAD_SIZE:-128}; PAGE_SIZE=${PAGE_SIZE:-16}
read -r ov fair < <(BE="$BACKEND" HS="$HEAD_SIZE" CS="$CS_H" PS="$PAGE_SIZE" \
    FP8="${FP8_PROJK:-0}" MAXOK="$conc_max" python3 -c '
import os
be=os.environ["BE"]; hs=int(os.environ["HS"]); cs=int(os.environ["CS"])
ps=int(os.environ["PS"]); fp8=os.environ["FP8"]=="1"; mx=int(os.environ["MAXOK"])
slot_b = 2*hs*2                       # K+V, bf16 (2 bytes)
# UNCOUNTED separate side-buffer bytes per token (0 if in-slot/none):
if be=="quest":       extra_b = (2*hs/ps)*2          # per-page minmax (bf16), amortized/token
elif be=="lrosa" and fp8: extra_b = cs*1             # fp8 proj_K (1 byte), separate contig cache
elif be=="lrosa":     extra_b = 0                    # bf16 proj_K is IN slot -> already counted
elif be=="seer":      extra_b = 128*2                # gate hidden (128), bf16, per token
else:                 extra_b = 0                    # fkv / fasa / lrosa_mla(see note)
f = slot_b/(slot_b+extra_b) if (slot_b+extra_b)>0 else 1.0
print(f"{(1-f)*100:.2f}", int(mx*f))
')
echo "SIDE_BUFFER_OVERHEAD_PCT=$ov  MEM_FAIR_CONCURRENT_MAX_BATCH=$fair  (= CONCURRENT_MAX_BATCH x slot/(slot+side_buffer); FKV/FASA/bf16-LRoSA overhead=0)" | tee -a "$summary"
echo "(full per-bsz logs in $OUT/${tag}_b*.log; summary -> $summary)"
