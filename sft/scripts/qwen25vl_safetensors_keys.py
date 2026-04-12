#!/usr/bin/env python3
"""
Qwen2.5-VL merged checkpoints sometimes save vision weights under wrong prefixes.
HuggingFace / vLLM expect top-level ``visual.*``. Common bad prefixes:

- ``model.visual.*`` (mis-nested under ``model``)
- ``base_model.model.visual.*`` (PEFT-style before unwrap)

Rename to ``visual.*`` in every *.safetensors shard.
"""

import argparse
import glob
import os
from collections import Counter
from typing import Any, Dict, List, Tuple

# Longest-first so we do not partially strip a shorter prefix first.
_RENAME_PREFIXES: List[Tuple[str, str]] = [
    ("base_model.model.visual.", "visual."),
    ("model.visual.", "visual."),
]


def normalize_qwen2_5_vl_state_dict_keys(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in state_dict.items():
        nk = k
        for bad, good in _RENAME_PREFIXES:
            if nk.startswith(bad):
                nk = good + nk[len(bad) :]
                break
        out[nk] = v
    return out


def _count_renames_in_keys(keys: List[str]) -> int:
    n = 0
    for k in keys:
        for bad, _ in _RENAME_PREFIXES:
            if k.startswith(bad):
                n += 1
                break
    return n


def fix_qwen25vl_visual_prefix_in_dir(
    export_dir: str,
    *,
    verbose: bool = False,
) -> tuple[int, int]:
    """
    In-place rewrite of *.safetensors under export_dir.
    Returns (files_modified, tensors_renamed).
    """
    try:
        from safetensors.torch import load_file, save_file
    except ImportError as e:
        raise ImportError("safetensors is required: pip install safetensors") from e

    paths = sorted(glob.glob(os.path.join(export_dir, "*.safetensors")))
    paths = [p for p in paths if "adapter" not in os.path.basename(p).lower()]
    if not paths:
        if verbose:
            print(f"[fix_qwen25vl_keys] No *.safetensors found under {export_dir!r}")
        return 0, 0

    n_files = 0
    n_tensors = 0
    for path in paths:
        sd = load_file(path)
        keys = list(sd.keys())
        renamed_in_file = _count_renames_in_keys(keys)
        if renamed_in_file == 0:
            continue
        new_sd = normalize_qwen2_5_vl_state_dict_keys(sd)
        save_file(new_sd, path)
        n_files += 1
        n_tensors += renamed_in_file

    if verbose and n_tensors == 0 and paths:
        sd0 = load_file(paths[0])
        keys0 = list(sd0.keys())
        print("[fix_qwen25vl_keys] No known bad vision prefixes; sample keys from first shard:")
        for k in keys0[:16]:
            print(f"  {k}")
        tops = Counter(k.split(".", 1)[0] for k in keys0)
        print(f"[fix_qwen25vl_keys] Top-level key prefixes in {os.path.basename(paths[0])}:")
        for pref, cnt in tops.most_common(24):
            print(f"  {pref!r}: {cnt}")

    return n_files, n_tensors


def main() -> None:
    p = argparse.ArgumentParser(
        description="Fix vision key prefixes for Qwen2.5-VL safetensors (HF / vLLM)"
    )
    p.add_argument("export_dir", type=str, help="Directory containing model *.safetensors")
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="If nothing renamed, print sample keys and prefix counts",
    )
    args = p.parse_args()
    d = os.path.abspath(args.export_dir)
    if not os.path.isdir(d):
        raise SystemExit(f"Not a directory: {d}")
    files, tensors = fix_qwen25vl_visual_prefix_in_dir(d, verbose=args.verbose)
    print(f"[fix_qwen25vl_keys] dir={d} files_modified={files} tensors_renamed={tensors}")
    if tensors == 0 and not args.verbose:
        print(
            "[fix_qwen25vl_keys] Hint: run with -v to see which key prefixes are in the checkpoint."
        )


if __name__ == "__main__":
    main()
