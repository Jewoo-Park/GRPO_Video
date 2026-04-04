import argparse
import json
import shutil
from pathlib import Path

from .utils import load_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge processed_<subset> directories into a single processed directory.")
    parser.add_argument("--root-dir", type=str, default="data/video_r1_cot")
    parser.add_argument("--input-pattern", type=str, default="processed_*")
    parser.add_argument("--output-name", type=str, default="processed")
    parser.add_argument("--processed-name", type=str, default="train.jsonl")
    parser.add_argument("--summary-name", type=str, default="merge_summary.json")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sample_key(row: dict) -> tuple[str, str, str, str]:
    return (
        str(row.get("source_subset") or "").strip(),
        str(row.get("video_path") or "").strip(),
        str(row.get("frame_subdir") or "").strip(),
        str(row.get("question") or "").strip(),
    )


def copy_frame_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        raise FileExistsError(f"Refusing to overwrite existing frame directory: {dst}")
    shutil.copytree(src, dst)


def main() -> None:
    args = parse_args()

    root_dir = Path(args.root_dir)
    output_dir = root_dir / args.output_name
    if output_dir.exists():
        if not args.force:
            raise SystemExit(
                f"Output directory already exists: {output_dir}. "
                "Remove it first or rerun with --force."
            )
        shutil.rmtree(output_dir)

    input_dirs = sorted(
        path for path in root_dir.glob(args.input_pattern)
        if path.is_dir() and path.name != args.output_name
    )
    if not input_dirs:
        raise SystemExit(f"No input directories matched {args.input_pattern!r} under {root_dir}")

    merged_rows_by_key: dict[tuple[str, str, str, str], dict] = {}
    copied_frame_subdirs: set[str] = set()
    input_stats: list[dict] = []
    duplicate_rows_skipped = 0

    output_frames_root = output_dir / "frames" / "train"
    output_frames_root.mkdir(parents=True, exist_ok=True)

    for input_dir in input_dirs:
        processed_path = input_dir / args.processed_name
        if not processed_path.exists():
            raise FileNotFoundError(f"Missing processed jsonl: {processed_path}")

        rows = load_jsonl(processed_path)
        input_stats.append({
            "input_dir": str(input_dir.resolve()),
            "rows": len(rows),
        })

        for row in rows:
            key = sample_key(row)
            if key in merged_rows_by_key:
                duplicate_rows_skipped += 1
                continue
            merged_rows_by_key[key] = row

            frame_subdir = str(row.get("frame_subdir") or "").strip()
            if not frame_subdir or frame_subdir in copied_frame_subdirs:
                continue

            src_frame_dir = input_dir / "frames" / "train" / frame_subdir
            if not src_frame_dir.exists():
                raise FileNotFoundError(f"Missing frame directory referenced by row: {src_frame_dir}")
            dst_frame_dir = output_frames_root / frame_subdir
            copy_frame_tree(src_frame_dir, dst_frame_dir)
            copied_frame_subdirs.add(frame_subdir)

    merged_rows = list(merged_rows_by_key.values())
    write_jsonl(output_dir / args.processed_name, merged_rows)

    summary = {
        "root_dir": str(root_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "inputs": input_stats,
        "input_directories": len(input_dirs),
        "merged_rows": len(merged_rows),
        "copied_frame_subdirs": len(copied_frame_subdirs),
        "duplicate_rows_skipped": duplicate_rows_skipped,
    }
    write_json(output_dir / args.summary_name, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
