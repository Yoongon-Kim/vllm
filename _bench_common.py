"""Shared helpers for the LRoSA / Quest vLLM benchmark harnesses.

Centralises the three things every harness (lrosa/quest smoke, qasper parity,
latency bench) needs to handle an arbitrary model rather than just Llama:

  * the pca research-repo path (LongBench prompts/metrics + calibrated bases),
  * model -> default basis-path resolution (pca's bases/<tag>/ naming), and
  * Qwen3 YaRN rope overrides that MUST mirror calibration so the LRoSA basis
    stays valid at long context.

Override the pca-repo location with the ``PCA_REPO`` env var if cloned
elsewhere; the default matches the current box.
"""
import os

# DeepSeek-V3.2-Exp declares model_type "deepseek_v32", which transformers 5.10
# doesn't know (it only registers deepseek_v3). vLLM has its own config registry,
# but the tokenizer's AutoTokenizer -> AutoConfig path uses RAW transformers and
# KeyErrors on "deepseek_v32". Register it as an alias of DeepseekV3Config (same
# config structure; DSA is driven by index_topk + the DeepseekV32ForCausalLM
# architecture, not the config class) so that path succeeds.
try:
    from transformers import AutoConfig
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
    from transformers.models.deepseek_v3.configuration_deepseek_v3 import (
        DeepseekV3Config as _DSV3Config,
    )

    if "deepseek_v32" not in CONFIG_MAPPING_NAMES:
        class DeepseekV32Config(_DSV3Config):  # noqa: D401
            model_type = "deepseek_v32"

        AutoConfig.register("deepseek_v32", DeepseekV32Config, exist_ok=True)
except Exception:  # transformers layout differs / already registered
    pass

# pca research repo (== this project's LRoSA-dev clone). Holds the LongBench
# v1 prompt/metric utils and the calibrated bases/<tag>/ trees.
PCA_REPO = os.environ.get("PCA_REPO", "/NHNHOME/jiwonsong/LRoSA-dev")


def model_tag(model: str) -> str:
    """pca's per-model bases subdir tag. Mirrors fasa/calibrate.py:750
    (lowercase, '-'/'.' -> '_', org prefix dropped):
        meta-llama/Llama-3.1-8B-Instruct -> llama_3_1_8b_instruct
        Qwen/Qwen3-8B                    -> qwen3_8b
    """
    return model.split("/")[-1].lower().replace("-", "_").replace(".", "_")


def lrosa_basis_path(model: str, cs_h: int = 32, variant: str = "d1") -> str:
    """Default pca basis path for ``model``:
        <PCA_REPO>/bases/<tag>/pca_<variant>_cs<N>_kv_head_<tag>.pt
    (D1 is the paper-default LRoSA basis; cs_h=32 for d=128 models.)
    """
    tag = model_tag(model)
    return os.path.join(
        PCA_REPO, "bases", tag,
        f"pca_{variant}_cs{cs_h}_kv_head_{tag}.pt",
    )


def fasa_idom_path(model: str) -> str:
    """Default pca FASA I_dom path for ``model``:
        <PCA_REPO>/bases/<tag>/fasa_idom_kv_head_<tag>.pt
    ({'idom': {layer: [H_kv, n_tip_max]}, ...}). Used by the paper-faithful
    fasa_fc backend mode (kv_cache_dtype="fasa").
    """
    tag = model_tag(model)
    return os.path.join(
        PCA_REPO, "bases", tag, f"fasa_idom_kv_head_{tag}.pt"
    )


def yarn_overrides(model: str) -> dict:
    """vLLM ``hf_overrides`` that enable YaRN rope for Qwen3 (factor=4 ->
    ~131K effective context), exactly mirroring fasa/calibrate.py's
    calibration-time rope so the LRoSA basis matches the served K distribution.

    transformers 5.x stores rope under ``rope_parameters`` (rope_theta nested),
    which is the key vLLM's hf_overrides expects (see
    examples/features/context_extension/context_extension_offline.py).

    Returns ``{}`` for non-Qwen3 models, or when the model config already
    carries a non-default rope (respect the user's extension).
    """
    if "qwen3" not in model.lower():
        return {}
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model)
    rp = getattr(cfg, "rope_parameters", None) or getattr(cfg, "rope_scaling", None) or {}
    if rp.get("rope_type", "default") != "default":
        return {}  # already extended -> leave it alone
    theta = rp.get("rope_theta", getattr(cfg, "rope_theta", None) or 1000000)
    return {
        "rope_parameters": {
            "rope_theta": theta,
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 32768,
        },
    }
