# Decode Throughput / Max-Batch Benchmark — Reproduction Guide

How to find the **max batch** that fits and measure **decode throughput**
(aggregate tokens/s) for LRoSA vs the baselines (FKV / FASA / Quest, and the
GLM-MLA LRoSA path), in the production vLLM stack (CUDA graph + the real
attention backend). Written so the process runs unchanged on a fresh GPU
server.

The bench is **single-node, one backend per process, decode-steady-state**.
It is NOT serving throughput (no request scheduling) — it times pure decode at
a fixed context and batch, which is the apples-to-apples sparse-attention
comparison.

---

## Status / what changed (read this first)

_Last updated 2026-06-22._

- **Use `MEM_FAIR_MAX_BATCH`, not the raw `MAX_BATCH`, to compare capacity across
  methods.** vLLM sizes `num_blocks` (max batch) from the per-token KV slot
  ONLY, so methods with a separate side-buffer that is NOT in the slot —
  **Quest** (`_quest_minmax`), **fp8/contig LRoSA proj_K** (`_projk_cache`),
  **Seer** (gate cache) — have that memory eat `gpu_mem` headroom instead of
  reducing `num_blocks`. Their raw `MAX_BATCH` is therefore **over-stated** (e.g.
  Quest's raw `MAX_BATCH` ≈ FKV's even though Quest uses more HBM). bf16 LRoSA
  proj_K is in-slot, so its `MAX_BATCH` is already fair.
- `sweep_max_batch_throughput.sh` now prints `SIDE_BUFFER_OVERHEAD_PCT` +
  `MEM_FAIR_MAX_BATCH = MAX_BATCH × slot/(slot+side_buffer)` (see §5). Overheads
  are symmetric where expected: Quest minmax **+5.88%** ≈ fp8-LRoSA proj_K
  **+5.88%**; FKV / FASA / bf16-LRoSA **0%**; Seer ≈ **33%**.
- New sweep env: `FP8_PROJK=1` (LRoSA fp8 contig proj_K), `HEAD_SIZE` /
  `PAGE_SIZE` (default 128 / 16 for Qwen3-8B).
- **vLLM's engine `num_blocks` is NOT modified** — this is a measurement-level
  correction (the bench reports the fair number). Making the engine itself
  size `num_blocks` correctly needs side-buffers carved from the raw KV
  allocation (`page_size_padded` + strided view) — a riskier per-backend
  refactor, deliberately deferred (the strided-view path is MLA-shape-only).
  Run the sweep at moderate `gpu_mem` (0.85–0.9) so the side-buffer sits in
  headroom and the fair scaling is clean.

---

## 0. What gets measured

`decode_latency_bench.py` runs two timed `generate()`s on `B` identical-length
random prompts (prefix-cache OFF, `ignore_eos`, CUDA graph on):

- `T_prefill` = time for `max_tokens=1` (prefill only)
- `T_full`   = time for `max_tokens=1+D` (prefill + `D` decode steps)

and prints (one line, grep-able):

```
DECODE_MS_PER_TOK=<ms>   PER_STREAM_TOK_S=<1000/ms>   AGG_TOK_S=<per_stream * B>
```

- **`DECODE_MS_PER_TOK`** = per-step decode latency (advances all `B` streams).
- **`AGG_TOK_S`** = aggregate decode throughput (tokens/s across the batch) ← the throughput number.
- **Max batch** = the largest `B` that does not OOM.
- **Peak throughput** = the highest `AGG_TOK_S` over the batch sweep (often, but
  not always, at max batch — decode is HBM-bound so throughput can plateau).

---

## 1. Prerequisites on the new server

1. **This vLLM fork**, built for the target GPU arch (B200 = `sm_100`). Decode
   bench needs the compiled `_C` + the custom LRoSA/Quest backends. If you only
   changed Python, a precompiled wheel install is fine; for kernel changes
   rebuild (`vllm/CLAUDE.md` / `HANDOFF.md`).
2. **Conda env** with vLLM + deps. Here it is `vllm`:
   `/home/snu_open/miniforge3/envs/vllm/bin/python` (override with `PY=`).
3. **Model weights** in the HF cache (`HF_HOME` / `HF_HUB_OFFLINE=1` for offline).
4. **Calibrated bases** for the sparse backends (`lrosa`/`fasa`/`lrosa_mla`):
   - Default lookup: `PCA_REPO/bases/<model_tag>/pca_d1_cs<CS_H>_kv_head_<tag>.pt`
     (LRoSA) and `.../fasa_idom_kv_head_<tag>.pt` (FASA). `PCA_REPO` defaults to
     `/NHNHOME/jiwonsong/LRoSA-dev` (override with the `PCA_REPO` env var).
   - **GLM-4.7-Flash bases live in `bases_la_full/glm_4_7_flash/`, NOT `bases/`**
     → pass `--basis` / `BASIS=` explicitly (see §4b).
   - `fkv` and `quest` need **no basis**.
5. **CUDA toolkit** at `/usr/local/cuda` (the GLM MoE flashinfer kernels JIT-compile
   and need `cuda_fp16.h` → `CUDA_HOME` + `CPATH`; the scripts export these).

Copy to the new server: the vLLM fork (built), the `LRoSA-dev/bases*` dirs, and
the model in the HF cache. Then set `PCA_REPO` / `HF_HOME` / `PY` to match.

---

## 2. Environment (exported by the scripts; shown for a manual run)

```bash
conda activate vllm                       # or use the absolute python via PY=
export HF_HOME=/path/to/hf_cache HF_HUB_OFFLINE=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0   # in-process engine (clean timing)
export VLLM_USE_FLASHINFER_SAMPLER=0      # native sampler (no ninja JIT)
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export CUDA_HOME=/usr/local/cuda PATH=$CUDA_HOME/bin:$PATH
export CPATH=$CUDA_HOME/targets/x86_64-linux/include${CPATH:+:$CPATH}
export PCA_REPO=/path/to/LRoSA-dev        # where bases/ lives
# IMPORTANT: a UNIQUE compile-cache dir per concurrent backend, else parallel
# procs racing one torch_compile_cache corrupt a kernel -> illegal mem access:
export VLLM_CACHE_ROOT=$HOME/.cache/vllm_<backend>
```

---

## 3. Single measurement (one backend, one context, one batch)

```bash
CUDA_VISIBLE_DEVICES=0 python decode_latency_bench.py \
  --backend lrosa --model Qwen/Qwen3-8B \
  --prefill_len 65536 --decode_len 128 --n_fac 2048 --cs_h 32 \
  --batch_size 8 --gpu_mem 0.9
# -> ... DECODE_MS_PER_TOK=14.9  PER_STREAM_TOK_S=67.1  AGG_TOK_S=536.9
```

Backends: `fkv` (dense upper bound), `lrosa`, `fasa`, `quest` (GQA models),
`lrosa_mla` (GLM-4.7-Flash MLA). `n_fac` = token budget (2048 = reasoning
default; 256 = LongBench default). For Qwen3 the bench **auto-applies YaRN when
context > 40k** (native ctx); short contexts run native.

---

## 4. Max-batch + throughput sweep (the main process)

`sweep_max_batch_throughput.sh` doubles `batch_size` (1,2,4,8,…) until OOM,
records `AGG_TOK_S` at each, and reports **MAX_BATCH** + **PEAK_THROUGHPUT**.

### 4a. GQA models (Qwen3-8B / Llama-3.1-8B): fkv / lrosa / fasa / quest

```bash
# one (backend, context) sweep on GPU 0:
GPU=0 BACKEND=lrosa CTX=65536 MODEL=Qwen/Qwen3-8B NFAC=2048 \
  bash sweep_max_batch_throughput.sh
# lrosa/fasa basis auto-resolves to PCA_REPO/bases/qwen3_8b/...; or pass BASIS=.

# full matrix (4 backends x 4 contexts), one backend per GPU per context round:
for ctx in 16384 32768 65536 130816; do
  gpu=0
  for be in fkv fasa lrosa quest; do
    GPU=$gpu BACKEND=$be CTX=$ctx MODEL=Qwen/Qwen3-8B NFAC=2048 \
      bash sweep_max_batch_throughput.sh &
    gpu=$((gpu+1))
  done
  wait
done
# Summaries: /NHNHOME/jiwonsong/tmp/tput_sweep/SUMMARY_<model>_<be>_c<ctx>.txt
```

### 4b. GLM-4.7-Flash (MLA): fkv vs lrosa_mla, fp8 latent KV

GLM needs the MLA flags + the `bases_la_full` basis. **`--chunked_prefill` is
required at ctx ≥ 32k** (the prefill workspace OOMs otherwise, so the real
decode batch can't fit — this is a prefill-only issue, decode timing is
unaffected).

```bash
GLM_BASIS=/NHNHOME/jiwonsong/LRoSA-dev/bases_la_full/glm_4_7_flash/pca_d1_cs64_ropeaware_kv_head_glm_4_7_flash.pt

# LRoSA-MLA (sparse):
GPU=0 BACKEND=lrosa_mla CTX=65536 MODEL=zai-org/GLM-4.7-Flash NFAC=2048 CS_H=64 \
  BASIS=$GLM_BASIS MLA_KV_DTYPE=fp8_ds_mla MLA_BACKEND=FLASHMLA_SPARSE \
  CHUNKED_PREFILL=1 GPU_MEM=0.9 \
  bash sweep_max_batch_throughput.sh

# FKV reference at the SAME fp8 latent KV (apples-to-apples dense vs sparse):
GPU=1 BACKEND=fkv CTX=65536 MODEL=zai-org/GLM-4.7-Flash NFAC=2048 \
  MLA_KV_DTYPE=fp8_ds_mla MLA_FKV_FP8=1 CHUNKED_PREFILL=1 GPU_MEM=0.9 \
  bash sweep_max_batch_throughput.sh
```

GLM gotchas (from prior runs): iso-fp8 LRoSA-MLA is ~1.5–2.6× FKV at ≤16k and
~3.4–3.9× at low batch out to 128k. **FKV-fp8 can hit a CUBLAS error at high
batch** — if the `fkv` sweep dies non-OOM at large `B`, that is the known
FKV-fp8 high-batch bug, not a real OOM (note it, cap `MAX_BSZ`).

> The throughput bench covers `lrosa_mla` (+ `fkv`) for GLM. `quest_mla` /
> `fasa_mla` / `triattn_mla` are accuracy-only (`reasoning_vllm_eval.py`) and
> are NOT in `decode_latency_bench.py`'s backend list.

---

## 5. Reading the results

Each sweep writes `SUMMARY_<tag>.txt`:

```
# bsz   decode_ms/tok   per_stream_tok/s   AGG_tok/s(throughput)   status
1       15.3            65.4               65.4                    OK
2       15.5            64.5               129.0                   OK
...
64      28.1            35.6               2275.0                  OK
128     -               -                  -                       OOM (stop)
----
MAX_BATCH=64  PEAK_THROUGHPUT_TOK_S=2275.0 @ bsz=64
SIDE_BUFFER_OVERHEAD_PCT=5.88  MEM_FAIR_MAX_BATCH=60  (= MAX_BATCH x slot/(slot+side_buffer); FKV/FASA/bf16-LRoSA overhead=0)
```

- **Max batch** = last `OK` row's `bsz`.
- **Peak throughput** = `PEAK_THROUGHPUT_TOK_S` (highest `AGG_tok/s`).
- **Speedup vs FKV** = backend `PEAK_THROUGHPUT_TOK_S` / FKV `PEAK_THROUGHPUT_TOK_S`
  at the same context (and/or compare `AGG_TOK_S` at a fixed batch).
- A `FAILED(non-OOM)` row → read its `*_b<bsz>.log` (could be the FKV-fp8 CUBLAS
  bug, a kernel/cache race — ensure unique `VLLM_CACHE_ROOT` — or a config error).

### Max-batch fairness — count the side-buffers (`MEM_FAIR_MAX_BATCH`)

vLLM sizes `num_blocks` (≈ max batch capacity) **only from the per-token KV
slot** (`real_page_size_bytes`). But several methods keep an extra buffer that
is NOT in that slot:

| method | side-buffer | in slot? | counted in `num_blocks`? |
|---|---|---|---|
| FKV / FASA | none | — | — (max batch is the dense ceiling) |
| **LRoSA bf16** proj_K | `[…,cs_h]` per token | **in slot** | ✅ yes → `MAX_BATCH` already fair |
| **LRoSA fp8** proj_K (contig) | separate fp8 cache | no | ❌ eats headroom |
| **Quest** | `_quest_minmax` per page | no | ❌ eats headroom |
| **Seer** | `_seer_kc` gate cache | no | ❌ eats headroom |

So for Quest / fp8-LRoSA / Seer the empirical `MAX_BATCH` is **over-stated** —
the side-buffer fits in `gpu_mem` headroom (1−gpu_mem) instead of reducing
`num_blocks`. That is why **Quest's raw MAX_BATCH ≈ FKV's** even though Quest
uses more HBM. `MEM_FAIR_MAX_BATCH` charges the side-buffer like the slot
(`MAX_BATCH × slot/(slot+side_buffer)`) so the comparison is iso-memory.
Notably Quest's minmax (`+5.88%`) ≈ fp8-LRoSA proj_K (`+5.88%`) — same real
overhead, now reported consistently. **Use `MEM_FAIR_MAX_BATCH` (not raw
`MAX_BATCH`) when comparing capacity across methods.** Set `HEAD_SIZE` /
`PAGE_SIZE` if not the Qwen3-8B defaults (128 / 16).

> Doing this in the engine instead (so vLLM's own `num_blocks` is correct)
> needs the side-buffers carved from the raw KV allocation via
> `page_size_padded` + a strided view — a real per-backend refactor (the
> strided-view path is currently MLA-shape-only). The measurement-level
> `MEM_FAIR_MAX_BATCH` gives the same fair number without that risk.

---

## 6. Knobs / tuning

| env | meaning | default |
|---|---|---|
| `GPU` | CUDA device | 0 |
| `BACKEND` | fkv\|lrosa\|fasa\|quest\|lrosa_mla | (required) |
| `CTX` | prefill_len (context) | (required) |
| `MODEL` | HF id | Qwen/Qwen3-8B |
| `NFAC` | token budget (n_fac) | 2048 |
| `CS_H` / `N_TIP` | LRoSA rank / FASA n_tip | 32 / 16 |
| `DECODE` | decode steps timed | 128 |
| `GPU_MEM` | gpu_memory_utilization | 0.9 |
| `BSZ_START` / `MAX_BSZ` | sweep bounds (doubling) | 1 / 512 |
| `BASIS` | basis .pt (req. for GLM; else auto from `PCA_REPO`) | — |
| `MLA_KV_DTYPE` / `MLA_BACKEND` | MLA latent KV / attn backend | — |
| `CHUNKED_PREFILL` | `--chunked_prefill` (MLA ctx≥32k) | 0 |
| `MLA_FKV_FP8` | fkv on MLA at fp8 latent KV | 0 |
| `PY` / `PCA_REPO` / `OUT` | python / bases root / results dir | env-set |

Notes:
- The doubling sweep gives a power-of-2 max batch. For an exact max, re-run a
  few linear `--batch_size` values between the last OK and the OOM batch.
- `decode_latency_bench.py` sets `max_num_seqs = max(batch_size, 1)` by default
  (tight buffers). Decode is steady-state; raise `gpu_mem` to push max batch.
- Keep `VLLM_CACHE_ROOT` unique per backend when running backends in parallel.
