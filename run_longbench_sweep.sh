#!/usr/bin/env bash
# LongBench v1 memory-vs-F1 SWEEP on the vLLM backends:
#   LRoSA  x cs_h  in {4,8,16,32}
#   FASA   x n_tip in {2,4,8,16}     (iso-byte pairing: n_tip = cs_h / 2)
#   FKV, Quest: budget-independent -> run once (reused from the cs32 table run
#               at /NHNHOME/jiwonsong/tmp/lb_table/<tag> if present).
#
# The LRoSA/FASA decode score+radix scratch is sized by max_num_seqs * max_kv.
# vLLM's default max_num_seqs=1024 makes it explode at 127500 ctx -> OOM in both
# CUDA-graph capture and inference, even though only ~5 seqs fit. Capping
# MAXSEQS (default 8) + gpu_mem 0.65 shrinks it so CUDA graphs capture AND decode
# fit (verified). Set EAGER=1 to disable graphs (F1 identical; graphs are a
# latency-only optimization).
#
# Usage:  MODEL=Qwen/Qwen3-14B bash run_longbench_sweep.sh
# Env:    MODEL CS_LIST NSAMP MAXLEN GPUMEM MAXSEQS EAGER TASKS GPUS HEARTBEAT NFAC
source "$HOME/miniforge3/etc/profile.d/conda.sh" && conda activate vllm
set -u
PY="${PY:-$HOME/miniforge3/envs/vllm/bin/python}"
cd /NHNHOME/jiwonsong/vllm
export TMPDIR=/NHNHOME/jiwonsong/tmp LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

MODEL=${MODEL:-Qwen/Qwen3-14B}
NFAC=${NFAC:-256}; NSAMP=${NSAMP:-200}; MAXLEN=${MAXLEN:-127500}
GPUMEM=${GPUMEM:-0.65}; TASKS=${TASKS:-all}; HEARTBEAT=${HEARTBEAT:-60}
MAXSEQS=${MAXSEQS:-8}        # cap concurrent seqs: bounds the max_kv-sized score
                            # /radix decode scratch so CUDA graphs capture AND
                            # decode fit at 127500 ctx (vLLM default 1024 OOMs).
EAGER=${EAGER:-0}           # 1 -> add --eager (graphs off); default keeps graphs.
read -r -a CS_LIST <<< "${CS_LIST:-4 8 16 32}"
read -r -a GPUS <<< "${GPUS:-0 1 2 3}"

if ! "$PY" -c "import vllm, fuzzywuzzy, rouge, transformers" 2>/tmp/_sw_dep.$$.err; then
  echo "ERROR: interpreter '$PY' is missing deps:"; cat /tmp/_sw_dep.$$.err
  echo "  fix: $PY -m pip install fuzzywuzzy python-Levenshtein rouge"
  rm -f /tmp/_sw_dep.$$.err; exit 1
fi
rm -f /tmp/_sw_dep.$$.err

TAG=$("$PY" -c "import _bench_common as b; print(b.model_tag('$MODEL'))")
NTASKS=$([ "$TASKS" = all ] && echo 16 || awk -F, '{print NF}' <<< "$TASKS")
OUT=/NHNHOME/jiwonsong/tmp/lb_sweep/$TAG; mkdir -p "$OUT"
echo "=== LongBench memory-vs-F1 sweep: $MODEL (tag=$TAG) ==="
echo "=== cs_h={${CS_LIST[*]}}  n_fac=$NFAC nsamp=$NSAMP maxlen=$MAXLEN gpu_mem=$GPUMEM eager ==="
echo "=== py=$PY  out=$OUT  $(date +%H:%M:%S) ==="

# Reuse FKV / Quest (budget-independent) from the cs32 table run if present.
SRC=/NHNHOME/jiwonsong/tmp/lb_table/$TAG
for m in fkv quest; do
  if [ ! -f "$OUT/$m.json" ] && [ -f "$SRC/$m.json" ]; then
    cp "$SRC/$m.json" "$OUT/$m.json"; echo "reused $m.json from $SRC"
  fi
done

# Job list: "label|mode|cs_h|n_tip". LRoSA(D1) + Loki(PCA-only) + FASA per cs_h,
# plus single FKV + Quest. Jobs whose json already exists are SKIPPED, so
# re-running the sweep only fills what's missing (e.g. adding Loki later).
all_jobs=()
for cs in "${CS_LIST[@]}"; do
  all_jobs+=("lrosa_cs${cs}|lrosa|${cs}|0")
  all_jobs+=("loki_cs${cs}|loki|${cs}|0")
  all_jobs+=("fasa_nt$((cs / 2))|fasa|0|$((cs / 2))")
done
all_jobs+=("fkv|fkv|0|0")
all_jobs+=("quest|quest|0|0")
jobs=()
for spec in "${all_jobs[@]}"; do
  label="${spec%%|*}"
  if [ -f "$OUT/$label.json" ]; then echo "  skip $label (json exists)"; else jobs+=("$spec"); fi
done
echo "=== ${#jobs[@]} jobs (of ${#all_jobs[@]}) across ${#GPUS[@]} GPUs ==="
[ "${#jobs[@]}" -eq 0 ] && { echo "nothing to run; aggregating existing."; }

run_one() {  # label mode cs nt gpu
  local label=$1 mode=$2 cs=$3 nt=$4 g=$5 extra=""
  [ "$mode" = lrosa ] && extra="--cs_h $cs"
  [ "$mode" = fasa ]  && extra="--n_tip $nt"
  [ "$EAGER" = 1 ]    && extra="$extra --eager"
  CUDA_VISIBLE_DEVICES=$g VLLM_CACHE_ROOT="$HOME/.cache/sw_${TAG}_${label}" \
    "$PY" longbench_vllm_eval.py --mode "$mode" --model "$MODEL" --tasks "$TASKS" \
      --num_samples "$NSAMP" --n_fac "$NFAC" --max_input_len "$MAXLEN" \
      --gpu_mem "$GPUMEM" --max_num_seqs "$MAXSEQS" $extra --output "$OUT/$label.json" \
      > "$OUT/$label.log" 2>&1
}

ng=${#GPUS[@]}
for ((w = 0; w < ${#jobs[@]}; w += ng)); do
  echo "--- wave $((w / ng + 1)) @ $(date +%H:%M:%S) ---"
  labels=(); pids=()
  for ((j = 0; j < ng && w + j < ${#jobs[@]}; j++)); do
    IFS='|' read -r label mode cs nt <<< "${jobs[w + j]}"
    g=${GPUS[j]}
    echo "  launch $label ($mode cs=$cs nt=$nt) -> GPU $g"
    run_one "$label" "$mode" "$cs" "$nt" "$g" &
    labels+=("$label"); pids+=($!)
  done
  while :; do
    alive=0; for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && alive=$((alive + 1)); done
    line=""; for l in "${labels[@]}"; do
      n=$(grep -c '\[LB\]' "$OUT/$l.log" 2>/dev/null || true); line+="$l=${n:-0}/$NTASKS "
    done
    echo "[$(date +%H:%M:%S)] $line(alive $alive/${#pids[@]})"
    [ "$alive" -eq 0 ] && break
    sleep "$HEARTBEAT"
  done
  wait
done
echo "=== all waves done $(date +%H:%M:%S) ==="

"$PY" - "$OUT" "${CS_LIST[*]}" <<'PYEOF' | tee "$OUT/sweep_table.txt"
import json, os, sys
d = sys.argv[1]; cs_list = [int(x) for x in sys.argv[2].split()]
def load(name):
    p = os.path.join(d, name + ".json")
    return json.load(open(p)) if os.path.exists(p) else None
def ov(j): return j["overall"] if j else float("nan")
fkv, quest = load("fkv"), load("quest")
base = ov(fkv)
print("\n=== LongBench v1 memory-vs-F1 (OVERALL F1) ===")
print("%-12s %9s %9s %9s %9s %9s" % ("budget", "FKV", "FASA", "Loki", "LRoSA", "Quest"))
for cs in cs_list:
    nt = cs // 2
    lr, fa, lo = load("lrosa_cs%d" % cs), load("fasa_nt%d" % nt), load("loki_cs%d" % cs)
    print("%-12s %9.2f %9.2f %9.2f %9.2f %9.2f" % (
        "cs%d/nt%d" % (cs, nt), ov(fkv), ov(fa), ov(lo), ov(lr), ov(quest)))
if base == base:  # not NaN
    print("\n=== as pct of FKV ===")
    for cs in cs_list:
        nt = cs // 2
        lr, fa, lo = load("lrosa_cs%d" % cs), load("fasa_nt%d" % nt), load("loki_cs%d" % cs)
        pf = lambda j: (100 * ov(j) / base) if j else float("nan")
        print("  cs%-2d/nt%-2d   FASA %6.1f   Loki %6.1f   LRoSA %6.1f" % (
            cs, nt, pf(fa), pf(lo), pf(lr)))
    print("  FKV 100.0   Quest %.1f  (budget-independent refs)" % (
        100 * ov(quest) / base if quest else float("nan")))
PYEOF
echo "=== sweep table -> $OUT/sweep_table.txt ; per-backend json+log in $OUT ==="
