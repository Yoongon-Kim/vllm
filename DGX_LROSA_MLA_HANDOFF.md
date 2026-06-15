# GLM LRoSA-MLA — B200 FlashMLA patch + DGX Spark (GB10/sm121) Triton port

Two separate things live here:
1. **B200 (sm100):** the FlashMLA int32/grid fix that unblocks GLM-MLA **64k** — already
   working (64k b4 = **2.32×** vs FKV), but it lives in a gitignored vendored dir, so it
   needs re-applying as a patch after any rebuild/reclone. See §1.
2. **DGX Spark (GB10/sm121):** FlashMLA cannot run there at all (it's compiled sm_90a/sm_100
   only), so GLM LRoSA-MLA needs an **arch-portable Triton sparse-MLA decode path**. This is
   the work to do on the DGX side. See §2.

Qwen3 / Gemma / GPT-OSS LRoSA already run on any arch (pure-Triton score/topk + indexed-attend),
so **only GLM (MLA) needs the DGX Triton work.**

---

## §1 — B200 FlashMLA int32/grid patch (persist + rebuild)

**Symptom (before fix):** GLM `lrosa_mla` at prefill ≥ ~33k throws
`[FlashMLA] Stride exceeds int32 limit: ...`, then (after that) a combine `CUDA invalid argument`.

**Root cause:** GLM's sparse **prefill** sends the whole sequence through the **decode**
interface with `s_q = prefill_len` (e.g. 65536). At that `s_q`:
- outer strides overflow int32: `q.stride(0)=s_q·h_q·d_qk=2.42e9`, `out.stride(0)` and
  `o_accum.stride(0)=s_q·h_q·d_v=2³¹` → `int64_stride_to_int` (csrc/api/common.h) throws;
- the combine kernel grids `gridDim.y = s_q = 65536 > 65535` (CUDA y/z limit) → invalid arg.

**Fix (4 files, see `flashmla_int32_64k_b200.patch`):** widen `stride_q_b` / `stride_o_b` /
`stride_o_accum_split` → `int64_t` in `SparseAttnDecodeParams` + `CombineParams` (csrc/params.h),
pass those strides through without `int64_stride_to_int` (csrc/api/sparse_decode.h:414/418/474),
use `make_stride_helper<int64_t>` for the OUT/o_accum TMA descriptors
(csrc/sm100/prefill/sparse/fwd_for_small_topk/head128/phase1.cuh:1026/1050), and swap the
combine grid so `s_q` is `gridDim.x` (csrc/smxx/decode/combine/combine.cu: blockIdx + dim3).
sm100 addressing is TMA (uint64 in HW) so the `s_q·inner` products stay int32-safe; only the
outer strides needed widening.

**The fix lives in `.deps/flashmla-src/` which is gitignored** (vLLM downloads it). To persist:

```bash
# after a fresh vLLM build / reclone (which re-downloads .deps/flashmla-src @ tag a6ec2ba):
cd /NHNHOME/jiwonsong/vllm
git -C .deps/flashmla-src apply ../../flashmla_int32_64k_b200.patch    # or: patch -p1 -d .deps/flashmla-src < flashmla_int32_64k_b200.patch
# then rebuild the flashmla extension (see memory: vllm-rebuild-conda-cuda):
#   conda activate vllm; cd build/cmake_fix; ninja _flashmla_C
#   cp _flashmla_C.abi3.so ../../vllm/_flashmla_C.abi3.so
```

**Verify:** `quest_latency_bench.py --backend lrosa_mla --model zai-org/GLM-4.7-Flash
--basis bases/glm_4_7_flash/pca_d1_cs64_kv_head_glm_4_7_flash.pt --cs_h 64 --n_fac 2048
--prefill_len 65536 --batch_size 4 --decode_len 128 --mla_backend FLASHMLA_SPARSE --gpu_mem 0.75`
→ should print `DECODE_MS_PER_TOK` (no throw). Got 10.089 ms/tok, 396.5 agg tok/s; FKV
(`--backend fkv --mla_backend TRITON_MLA`) = 170.9 → **2.32×**. (32k was 2.14× → scales with ctx.)
Upstreaming to vllm-project/FlashMLA would remove the re-apply step.

---

## §2 — DGX Spark (GB10 / sm121): Triton sparse-MLA decode (the TODO)

**Why:** `cmake/external_projects/flashmla.cmake` builds FlashMLA for `sm_90a` + `sm_100` ONLY.
GB10 is sm_121 — sm_100a SASS is NOT forward-compatible, no PTX fallback → FlashMLA kernels
won't load on DGX. So GLM `lrosa_mla` (which uses `FLASHMLA_SPARSE`) cannot run on DGX as-is.
Need an arch-portable (Triton) sparse-MLA **decode** kernel.

**Goal:** reproduce FlashMLA `sparse_decode` semantics in Triton so GLM-4.7-Flash LRoSA-MLA
runs on DGX. Only the **attend** needs porting — the LRoSA **score + top-k selection** is
already pure Triton (`vllm/v1/attention/ops/triton_lrosa_score_topk.py`), arch-portable.

**Semantics to replicate (per decode step):**
- inputs: absorbed query `q [num_decode, h_q, 576]` (576 = 512 NoPE + 64 RoPE), paged latent
  cache `c_KV [num_blocks, block_size, 1, 656]` fp8_ds_mla (656 = 512 fp8 + 16 scale + 128 rope
  bf16; MQA = a single shared kv-head on the 576-dim latent), and per-request top-k indices
  `top_idx [num_decode, n_fac]` into the latent cache.
- compute: gather the `n_fac` selected latent rows, upconvert fp8→bf16, flash-style online
  softmax of `q·c_KV` over the n_fac tokens, output `[num_decode, h_q, 512]` (V = first 512 of latent).
- scale = 1/sqrt(576); causal not needed (selection is already causal-masked at score time).

**Starting points (in this repo):**
- WIP Triton MLA indexer: `vllm/model_executor/layers/lrosa_mla_indexer.py` — evaluate / finish.
- Mirror the dense-LRoSA gather-free attend: `vllm/v1/attention/ops/triton_lrosa_indexed_attend.py`
  (online-softmax `tl.dot`, int64 slot addressing, split-N for low-batch occupancy). Adapt for:
  single kv-head, 576-dim latent, fp8 upconvert, V = nope-512 + the rope handling.
- Backend wiring: `vllm/v1/attention/backends/mla/flashmla_sparse.py` currently calls
  `flash_mla_sparse_fwd` / `flash_mla_with_kvcache`. Add an **arch gate**: if not (sm90/sm100),
  dispatch decode (and the prefill-as-big-decode path) to the Triton kernel instead.
- fp8 upconvert reference: `ops.cp_gather_and_upconvert_fp8_kv_cache` (used in the prefill path).

**Config / basis (same as B200):** `--cs_h 64` (NOT 128 — GLM groups 64 latent + 64 rope = 128),
`--n_fac 2048`, fp8_ds_mla cache. Basis lives in LRoSA-dev: `bases/glm_4_7_flash/pca_d1_cs64_kv_head_glm_4_7_flash.pt`.

**Lessons from the B200 FlashMLA fix (apply to the Triton kernel from the start):**
- the long-prefill path runs the **whole sequence as one "decode" with s_q = prefill_len**, so
  the kernel must handle large s_q: use **int64 offsets** for any `row_idx * stride`, and if you
  grid over s_q, put it in a dim that allows > 65535 (gridDim.x), or tile it. Don't bake int32 /
  65535 assumptions (that's exactly what broke FlashMLA at 64k).

**Build on DGX:** rebuild vLLM `_C` for sm_121 (see memory `vllm-rebuild-conda-cuda`: conda-CUDA
`targets/` symlink fix + `ninja _C`; add `-gencode arch=compute_121,code=sm_121` for GB10). The
Triton kernels JIT at runtime per-arch, so they need no gencode — only vLLM's `_C` does.

**Test on DGX:**
- speed: `quest_latency_bench.py --backend lrosa_mla --model zai-org/GLM-4.7-Flash --cs_h 64
  --n_fac 2048 --basis <cs64> --prefill_len {4k,8k,16k,32k,64k} --batch_size 4 --mla_backend <triton-tag>`
- accuracy: `reasoning_vllm_eval.py --eval aime25 --mode lrosa_mla --model zai-org/GLM-4.7-Flash
  --basis <cs64> --cs_h 64 --n_fac 2048` (cross-check vs B200 AIME LongAlign GLM = 0.804 ≈ FKV).
- DGX is the most bandwidth-bound HW (GB10 ~273 GB/s) → expect the **largest** LRoSA-MLA speedup
  (B200 64k was 2.32× and B200 is bandwidth-rich; DGX should exceed that).

**Other models on DGX (no porting needed):** Qwen3/Gemma/GPT-OSS LRoSA use the pure-Triton
score/topk + `triton_lrosa_indexed_attend` (head ≤ 512), arch-portable — just rebuild vLLM `_C`
for sm_121 and run `quest_latency_bench.py --indexed_attend` / `reasoning_vllm_eval.py --mode lrosa`.
