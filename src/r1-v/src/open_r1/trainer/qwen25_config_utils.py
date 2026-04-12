# Small helpers for Qwen2.5-VL configs (merged checkpoints often omit fields).
import logging
import os
from typing import Any

from transformers import AutoConfig

logger = logging.getLogger(__name__)

# Qwen2.5-VL-7B-Instruct (and same-arch merges) expect multimodal RoPE sections.
_DEFAULT_QWEN25_MROPE = {"type": "mrope", "mrope_section": [16, 24, 24]}


def ensure_qwen25_rope_scaling(cfg: Any) -> None:
    """
    Merged LoRA exports sometimes save `rope_scaling: null`. Qwen2.5-VL decoder attention
    indexes `self.rope_scaling[\"mrope_section\"]` and crashes otherwise.
    Prefer copying from QWEN_BASE_PATH / PROCESSOR_PATH when set.
    """
    if cfg is None or getattr(cfg, "model_type", None) != "qwen2_5_vl":
        return
    rs = getattr(cfg, "rope_scaling", None)
    if isinstance(rs, dict) and rs.get("mrope_section") is not None:
        return

    base = (
        os.environ.get("VLLM_REFERENCE_CONFIG_PATH", "").strip()
        or os.environ.get("QWEN_BASE_PATH", "").strip()
        or os.environ.get("PROCESSOR_PATH", "").strip()
    )
    picked = None
    if base and os.path.isdir(base):
        try:
            bc = AutoConfig.from_pretrained(base, trust_remote_code=True)
            picked = getattr(bc, "rope_scaling", None)
            if isinstance(picked, dict) and picked.get("mrope_section") is not None:
                setattr(cfg, "rope_scaling", picked)
                logger.info(
                    "Filled missing rope_scaling from base config at %s",
                    base,
                )
                return
        except Exception as exc:
            logger.warning("Could not load rope_scaling from %s: %s", base, exc)

    setattr(cfg, "rope_scaling", _DEFAULT_QWEN25_MROPE)
    logger.warning(
        "Qwen2.5-VL config missing rope_scaling; using default %s. "
        "Prefer setting QWEN_BASE_PATH to a clean HF Qwen2.5-VL directory or fixing merged config.json.",
        _DEFAULT_QWEN25_MROPE,
    )
