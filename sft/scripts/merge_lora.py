#!/usr/bin/env python3
import argparse
import os
import shutil
import sys
import tempfile

import torch
import yaml
from peft import PeftModel
from transformers import AutoProcessor, AutoTokenizer

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
from qwen25vl_safetensors_keys import fix_qwen25vl_visual_prefix_in_dir


def remap_adapter_keys_and_prepare_dir(adapter_name_or_path: str) -> str:
    """
    Remap known adapter key mismatches (e.g. language_model.layers, visual.blocks,
    visual.merger, and default-adapter naming) so PeftModel.from_pretrained can
    load without 'missing adapter keys' warnings.
    Writes remapped adapter to a temp dir and returns that path.
    """
    try:
        from safetensors.torch import load_file, save_file
    except ImportError:
        raise ImportError("safetensors is required for remap_adapter_keys. pip install safetensors")
    adapter_path = os.path.abspath(adapter_name_or_path)
    safetensors_path = os.path.join(adapter_path, "adapter_model.safetensors")
    if not os.path.isfile(safetensors_path):
        return adapter_name_or_path
    sd = load_file(safetensors_path)
    new_sd = {}
    for k, v in sd.items():
        nk = k.replace(".model.model.language_model.layers.", ".model.model.layers.")
        nk = nk.replace(".model.model.visual.blocks.", ".model.visual.blocks.")
        # Same extra `.model.` wrapper as blocks; PEFT expects base_model.model.visual.merger.*
        nk = nk.replace(".model.model.visual.merger.", ".model.visual.merger.")
        # Training saves lora_A.weight; Peft default adapter uses lora_A.default.weight
        if ".visual.merger." in nk:
            nk = nk.replace("lora_A.weight", "lora_A.default.weight")
            nk = nk.replace("lora_B.weight", "lora_B.default.weight")
        new_sd[nk] = v
    tmpdir = tempfile.mkdtemp(prefix="merge_lora_remap_")
    try:
        save_file(new_sd, os.path.join(tmpdir, "adapter_model.safetensors"))
        for fn in ("adapter_config.json", "README.md"):
            src = os.path.join(adapter_path, fn)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(tmpdir, fn))
        return tmpdir
    except Exception:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--config", type=str, default=None, help="Path to merge config YAML")
    parser.add_argument("--model-name-or-path", type=str, default=None, help="Base model or merged SFT model path")
    parser.add_argument("--adapter-name-or-path", type=str, default=None, help="LoRA adapter directory")
    parser.add_argument("--export-dir", type=str, default=None, help="Directory to save merged weights")
    parser.add_argument(
        "--remap-adapter-keys",
        type=str,
        default=None,
        help="Override config and remap adapter keys before merge (true/false).",
    )
    return parser.parse_args()


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_processor_or_tokenizer(model_name_or_path: str, export_dir: str) -> None:
    try:
        processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
        processor.save_pretrained(export_dir)
        return
    except Exception:
        pass

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer.save_pretrained(export_dir)


def ensure_cuda_home() -> None:
    """Best-effort CUDA_HOME for tools that import DeepSpeed (merge save path avoids DS when possible)."""
    ch = os.environ.get("CUDA_HOME", "").strip()
    if ch and os.path.isdir(ch):
        return
    try:
        import torch.utils.cpp_extension as cep

        th = getattr(cep, "CUDA_HOME", None)
        if th and isinstance(th, str) and os.path.isdir(th):
            os.environ["CUDA_HOME"] = th
            return
    except Exception:
        pass
    for candidate in ("/usr/local/cuda", "/usr/local/cuda-12", "/usr/local/cuda-12.4"):
        if os.path.isdir(candidate):
            os.environ["CUDA_HOME"] = candidate
            return


def save_merged_pretrained(model, export_dir: str) -> None:
    """merge_and_unload() returns an unwrapped model; skip accelerate unwrap to avoid importing deepspeed."""
    import transformers.modeling_utils as modeling_utils

    _unwrap = modeling_utils.unwrap_model

    def _unwrap_identity(m, *args, **kwargs):
        return m

    try:
        modeling_utils.unwrap_model = _unwrap_identity  # type: ignore[assignment]
        model.save_pretrained(export_dir, safe_serialization=True)
    finally:
        modeling_utils.unwrap_model = _unwrap


def get_base_model(model_name_or_path: str):
    """Load Qwen2.5-VL base weights. No CausalLM fallback — VL config is incompatible and hides real errors."""
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration  # type: ignore
    except ImportError as e:
        raise ImportError(
            "Install a transformers build that provides Qwen2_5_VLForConditionalGeneration "
            "(Qwen2.5-VL merge is not supported via AutoModelForCausalLM)."
        ) from e
    return Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )


def main() -> None:
    args = parse_args()
    ensure_cuda_home()
    if args.config is not None:
        cfg = load_yaml(args.config)
    else:
        cfg = {}

    model_name_or_path = args.model_name_or_path or cfg.get("model_name_or_path")
    adapter_name_or_path = args.adapter_name_or_path or cfg.get("adapter_name_or_path")
    export_dir = args.export_dir or cfg.get("export_dir")
    remap_adapter_keys = cfg.get("remap_adapter_keys", False)
    if args.remap_adapter_keys is not None:
        remap_adapter_keys = str(args.remap_adapter_keys).strip().lower() in {"1", "true", "yes", "y", "on"}

    if not model_name_or_path or not adapter_name_or_path or not export_dir:
        raise ValueError(
            "model_name_or_path, adapter_name_or_path, and export_dir must be provided "
            "either via --config or direct CLI flags."
        )

    os.makedirs(export_dir, exist_ok=True)

    adapter_to_remove = None
    if remap_adapter_keys:
        adapter_name_or_path = remap_adapter_keys_and_prepare_dir(adapter_name_or_path)
        adapter_to_remove = adapter_name_or_path

    base_model = get_base_model(model_name_or_path)
    peft_model = PeftModel.from_pretrained(base_model, adapter_name_or_path)
    merged_model = peft_model.merge_and_unload()

    save_merged_pretrained(merged_model, export_dir)
    n_files, n_tensors = fix_qwen25vl_visual_prefix_in_dir(export_dir)
    if n_tensors:
        print(
            f"[merge_lora] Fixed HF/vLLM keys: model.visual.* -> visual.* "
            f"({n_tensors} tensors in {n_files} shard file(s))"
        )
    save_processor_or_tokenizer(model_name_or_path, export_dir)

    if adapter_to_remove and os.path.isdir(adapter_to_remove):
        shutil.rmtree(adapter_to_remove, ignore_errors=True)


if __name__ == "__main__":
    main()
