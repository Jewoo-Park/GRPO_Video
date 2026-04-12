#!/usr/bin/env python3
"""
Deterministically split a JSONL dataset into train/eval files without loading it all in memory.

Typical use (repo root):
  python src/scripts/split_jsonl_train_eval.py \
    --input data/video_r1/grpo/video_r1_grpo_train.jsonl \
    --train-out data/video_r1/grpo/splits/video_r1_grpo_train__train.jsonl \
    --eval-out  data/video_r1/grpo/splits/video_r1_grpo_train__eval.jsonl \
    --eval-fraction 0.02 \
    --seed 42 \
    --video-only true
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def parse_bool(s: str) -> bool:
    v = str(s).strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool: {s}")


def stable_key(obj: Dict[str, Any]) -> str:
    # Prefer stable identifiers if present.
    qid = obj.get("question_id")
    vid = obj.get("video_id")
    if qid is not None and vid is not None:
        return f"{vid}::{qid}"
    if vid is not None:
        return str(vid)
    if qid is not None:
        return str(qid)
    # Fallback: hash a normalized JSON dump of the object.
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


def looks_like_image_path(p: str) -> bool:
    pl = p.strip().lower()
    return pl.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif"))


def should_keep_video_only(obj: Dict[str, Any]) -> bool:
    vid = obj.get("video_id")
    if isinstance(vid, str) and looks_like_image_path(vid):
        return False
    return True


def bucket_for(key: str, seed: int) -> int:
    h = hashlib.sha256(f"{seed}::{key}".encode("utf-8")).hexdigest()
    return int(h[:8], 16)  # 32-bit bucket


def split_decision(key: str, eval_fraction: float, seed: int) -> bool:
    # True -> eval, False -> train
    b = bucket_for(key, seed)
    return (b / 0xFFFFFFFF) < eval_fraction


def count_lines(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--train-out", required=True)
    ap.add_argument("--eval-out", required=True)
    ap.add_argument("--eval-fraction", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--video-only", type=parse_bool, default=True)
    args = ap.parse_args()

    if not (0.0 < args.eval_fraction < 1.0):
        raise SystemExit("--eval-fraction must be in (0, 1)")

    in_path = Path(args.input).expanduser().resolve()
    train_out = Path(args.train_out).expanduser().resolve()
    eval_out = Path(args.eval_out).expanduser().resolve()

    train_out.parent.mkdir(parents=True, exist_ok=True)
    eval_out.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    kept_train = 0
    kept_eval = 0
    skipped_blank = 0
    skipped_parse = 0
    skipped_video_only = 0

    with in_path.open("r", encoding="utf-8") as fin, train_out.open("w", encoding="utf-8") as ftrain, eval_out.open(
        "w", encoding="utf-8"
    ) as feval:
        for line in fin:
            raw = line.strip()
            if not raw:
                skipped_blank += 1
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                skipped_parse += 1
                continue
            if not isinstance(obj, dict):
                skipped_parse += 1
                continue
            if args.video_only and not should_keep_video_only(obj):
                skipped_video_only += 1
                continue

            key = stable_key(obj)
            is_eval = split_decision(key, args.eval_fraction, args.seed)
            out_line = json.dumps(obj, ensure_ascii=False)
            if is_eval:
                feval.write(out_line + "\n")
                kept_eval += 1
            else:
                ftrain.write(out_line + "\n")
                kept_train += 1
            kept += 1

    # Best-effort fsync to reduce partial-write surprises on shared FS.
    try:
        os.sync()
    except Exception:
        pass

    print("[split] input:", str(in_path))
    print("[split] train_out:", str(train_out))
    print("[split] eval_out:", str(eval_out))
    print("[split] eval_fraction:", args.eval_fraction, "seed:", args.seed, "video_only:", args.video_only)
    print(
        "[split] kept_total:",
        kept,
        "kept_train:",
        kept_train,
        "kept_eval:",
        kept_eval,
        "skipped_video_only:",
        skipped_video_only,
        "skipped_blank:",
        skipped_blank,
        "skipped_parse:",
        skipped_parse,
    )

    # sanity: ensure eval isn't empty
    if kept_eval == 0:
        raise SystemExit("[split] ERROR: eval set is empty; increase --eval-fraction or check keys")


if __name__ == "__main__":
    main()

