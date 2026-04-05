import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from .utils import load_jsonl, write_json


DEFAULT_QUOTAS_4K = {
    "LLaVA-Video-178K": 1400,
    "Knowledge": 500,
    "Math": 450,
    "Chart": 300,
    "Spatial": 250,
    "General": 180,
    "STAR": 300,
    "NeXT-QA": 220,
    "OCR": 180,
    "CLEVRER": 150,
    "PerceptionTest": 70,
}

DEFAULT_QUOTAS_6K = {
    "LLaVA-Video-178K": 2100,
    "Knowledge": 750,
    "Math": 675,
    "Chart": 450,
    "Spatial": 375,
    "General": 270,
    "STAR": 450,
    "NeXT-QA": 330,
    "OCR": 270,
    "CLEVRER": 225,
    "PerceptionTest": 105,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample a smaller processed dataset with fixed subset quotas.")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preset", type=str, default="balanced_4k")
    parser.add_argument(
        "--exclude-dir",
        type=str,
        default=None,
        help="Path to a previously sampled output dir. Rows already sampled there will be excluded.",
    )
    return parser.parse_args()


def quotas_for_preset(name: str) -> dict[str, int]:
    if name == "balanced_4k":
        return dict(DEFAULT_QUOTAS_4K)
    if name == "additional_6k":
        return dict(DEFAULT_QUOTAS_6K)
    raise ValueError(f"Unsupported preset: {name}")


def load_excluded_ids(exclude_dir: Path) -> set[str]:
    train_path = exclude_dir / "train.jsonl"
    if not train_path.exists():
        raise SystemExit(f"--exclude-dir given but train.jsonl not found: {train_path}")
    rows = load_jsonl(train_path)
    excluded: set[str] = set()
    for row in rows:
        qid = str(row.get("question_id") or "").strip()
        if qid:
            excluded.add(qid)
    return excluded


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    quotas = quotas_for_preset(args.preset)
    rng = random.Random(args.seed)

    train_path = input_dir / "train.jsonl"
    if not train_path.exists():
        raise SystemExit(f"Missing input file: {train_path}")

    excluded_ids: set[str] = set()
    if args.exclude_dir:
        excluded_ids = load_excluded_ids(Path(args.exclude_dir))
        print(f"[sample] excluding {len(excluded_ids)} already-sampled question_ids from {args.exclude_dir}")

    rows = load_jsonl(train_path)
    if excluded_ids:
        before = len(rows)
        rows = [r for r in rows if str(r.get("question_id") or "").strip() not in excluded_ids]
        print(f"[sample] filtered {before} → {len(rows)} rows after exclusion")

    rows_by_subset: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        rows_by_subset[str(row.get("source_subset") or "UNKNOWN")].append(row)

    missing_subsets = [subset for subset in quotas if subset not in rows_by_subset]
    if missing_subsets:
        raise SystemExit(f"Missing subsets in input: {missing_subsets}")

    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    sampled_rows: list[dict] = []
    sampling_stats: dict[str, dict[str, int]] = {}

    for subset, sample_size in quotas.items():
        pool = rows_by_subset[subset]
        if len(pool) < sample_size:
            raise SystemExit(
                f"Subset {subset} has only {len(pool)} rows, cannot sample requested {sample_size}"
            )
        picked = rng.sample(pool, sample_size)
        sampled_rows.extend(picked)
        sampling_stats[subset] = {
            "total": len(pool),
            "sampled": sample_size,
        }

    rng.shuffle(sampled_rows)

    with (output_dir / "train.jsonl").open("w", encoding="utf-8") as f:
        for row in sampled_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    unique_frame_subdirs = {
        str(row.get("frame_subdir") or "").strip()
        for row in sampled_rows
        if str(row.get("frame_subdir") or "").strip()
    }
    for frame_subdir in sorted(unique_frame_subdirs):
        src_dir = input_dir / "frames" / "train" / frame_subdir
        if not src_dir.exists():
            raise SystemExit(f"Missing frame directory for sampled row: {src_dir}")

    src_frames_dir = (input_dir / "frames").resolve()
    dst_frames_link = output_dir / "frames"
    if dst_frames_link.exists() or dst_frames_link.is_symlink():
        dst_frames_link.unlink()
    dst_frames_link.symlink_to(src_frames_dir, target_is_directory=True)

    summary = {
        "preset": args.preset,
        "seed": args.seed,
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "exclude_dir": args.exclude_dir,
        "excluded_ids": len(excluded_ids),
        "input_rows_after_exclusion": len(rows),
        "sampled_rows": len(sampled_rows),
        "linked_frames_dir": str(src_frames_dir),
        "unique_frame_subdirs": len(unique_frame_subdirs),
        "sampling_stats": sampling_stats,
    }
    write_json(output_dir / "sample_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
