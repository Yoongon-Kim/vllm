# GLM (MLA) FASA-MLA & QUEST-MLA — implementation spec

## 0. Architecture / current state
GLM-4.7-Flash MLA per token: **NoPE latent c_KV (kv_lora_rank=512) + decoupled RoPE k_pe (qk_rope_head_dim=64 → 32 RoPE FCs)**; H=20 heads, 47 layers.
MLA decode score = `q_latent·c_KV` (q_latent = q_nope@W_UK, 512-d) `+ q_pe·k_pe` (64-d).

Sparse-decode infra (already built, reused by all variants):
`Indexer.forward → builds index_q[H,hd], index_k[hd] → SparseAttnIndexer (fp8 paged-MQA-logits + CTA radix top-k) → topk_indices_buffer(n_fac) → FlashMLASparse attend`.
Hook = `vllm/model_executor/layers/lrosa_mla_indexer.py::LRoSAMLAIndexer` (built in glm4_moe_lite.py / deepseek_v2.py, gated by `attention_config.lrosa_mla`).
- **LRoSA-MLA** index_q[h]=`[M·q_latent[h] | q_pe[h]]`, index_k=`[M·c_KV | k_pe]` (head_dim=cs_h+64=128). Score = (M q_latent)·(M c_KV)+q_pe·k_pe.
- **Loki-MLA** = same with q-blind PCA M.
- Kernel constraint: deepgemm fp8 logits supports H∈{32,64} (H=20 zero-padded) and head_dim is one fp8 quant block (128); ReLU-shift trick (`shift_c`) makes RELU(q·k) == head-sum.

## 1. FASA-MLA  (LOW-MODERATE — variant of LRoSAMLAIndexer, reuses the fp8 kernel)
FASA scores ONLY the dominant decoupled-RoPE FCs (I_dom ⊂ 32 RoPE FCs); the NoPE latent has no RoPE → not FASA-scored. This is exactly the paper's **partial-RoPE** recipe (DeepSeek-V2-Lite rebuttal, Fig.11 App A.1; FASA≈FKV at N_fac256 on LongBench).
- New `FASAMLAIndexer` (clone of LRoSAMLAIndexer) with index build replaced by:
  - `index_q[h] = [ mask_{I_dom}(q_pe[h])  (64) | 0 (64) ]`  (zero the non-I_dom RoPE-pair channels)
  - `index_k    = [ k_pe                   (64) | 0 (64) ]`
  - head_dim=128 (kernel-compatible, 64 zero-pad); score = Σ_{i∈I_dom} (q_pe·k_pe)_i. NO M, NO c_KV.
- Calibration: extend `_install_recording_forward_glm4_moe_lite` to ALSO record post-RoPE `q_pe[H,64]` + `k_pe[64]` (currently records c_KV+q_latent only). Compute FASA CA per RoPE-FC (2-dim) over the 32 FCs → I_dom = top-N_tip FCs (reuse the existing `_compute_idom_from_ca`, just feed RoPE-FC CA). Save `fasa_idom_mla_*` (per-layer 32-FC dominant mask).
- Config: `attention_config.fasa_mla=True`, `lrosa_n_tip` = #FCs. glm4_moe_lite picks FASAMLAIndexer when set.
- NOTE: masking (not gather) keeps fp8 per-128-block quant valid; padded/zeroed dims add 0 to the logit.

## 2. QUEST-MLA  (MODERATE-HIGHER — needs a NON-dot-product scorer)
Quest = page-level upper bound `Σ_c max(q[c]·Kmin[c], q[c]·Kmax[c])` per page — min/max, NOT a q·k dot product → **cannot reuse SparseAttnIndexer's fp8 logits kernel**.
- New `QuestMLAIndexer`: on the absorbed key `[q_latent | q_pe]` vs `[c_KV | k_pe]` (576-d), maintain per-page (page=16) per-channel min/max of `[c_KV | k_pe]`; decode score per page = Σ_c max(q·Kmin, q·Kmax); top-(n_fac/16) pages → expand to token indices → write topk_indices_buffer → FlashMLASparse (unchanged).
- Impl: PyTorch eager page min/max scorer (accuracy eval, not throughput — eager is fine; matches our transformers Quest port which is also pure-PyTorch). Page min/max maintained over the c_KV+k_pe cache (recompute from the latent cache, or a running buffer).
- No calibration (Quest is calibration-free). Config: `attention_config.quest_mla=True`, page_size=16, budget=lrosa_n_fac.
- Risk: token-vs-page granularity into FlashMLASparse (which wants token indices). Expand selected pages → 16 token ids each, cap at n_fac. Partial last page handled like the GQA Quest port.

## 3. Integration points (files)
- `lrosa_mla_indexer.py`: add FASAMLAIndexer (+ QuestMLAIndexer or a sibling file).
- `glm4_moe_lite.py` / `deepseek_v2.py`: indexer dispatch on `fasa_mla` / `quest_mla` (mirror the `lrosa_mla` branch at L226).
- `config/attention.py`: add `fasa_mla`, `quest_mla` bool flags.
- `fasa/calibrate.py`: record q_pe/k_pe in glm4_moe_lite recording + RoPE-FC CA → `fasa_idom_mla` basis (FASA-MLA only).
- `reasoning_vllm_eval.py` / `ruler_vllm_eval.py` / `longbench_vllm_eval.py`: add `--mode fasa_mla|quest_mla` (set the attention_config flag + basis path for fasa).

## 4. Order & validation
1. FASA-MLA: calib q_pe/k_pe I_dom → FASAMLAIndexer → smoke (load+generate) → reasoning/LongBench/RULER.
2. QUEST-MLA: QuestMLAIndexer (eager page-min/max) → smoke → same evals.
- Completes GLM baseline set: FKV / LRoSA-MLA / Loki-MLA / **FASA-MLA / QUEST-MLA**.

## 5. Caveat (honest)
FASA-MLA scores the decoupled-RoPE part only (positional, 32 FCs); LRoSA scores the NoPE latent. On LongBench/reasoning FASA-MLA should ≈FKV (paper's DeepSeek-V2 result). On RULER multikey, MLA latent retrieval was weak for LRoSA/Loki — FASA(RoPE)/Quest(page) may behave differently but could also be limited; this is the thing to measure, not assume.
