import argparse
import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset

from .utils import (
    download_dataset_files,
    extract_archives,
    extract_frames_for_rows,
    hf_token_from_env,
    is_video_row,
    normalize_answer_text,
    pick_gold_answer,
    pick_options,
    pick_question,
    sample_rows_by_subset,
    write_json,
    write_jsonl,
)


def normalize_repo_path(path_text: str) -> str:
    return path_text.strip().lstrip("./")


def match_subset(row: dict, subsets: list[str] | None) -> str:
    path_value = normalize_repo_path(str(row.get("path") or ""))
    source_value = str(row.get("data_source") or "").strip()

    if subsets:
        for subset in subsets:
            if source_value == subset or path_value.startswith(f"{subset}/"):
                return subset
        return ""

    if source_value:
        return source_value
    if path_value:
        return path_value.split("/", 1)[0]
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Video-R1-COT-165k video rows for annotation.")
    parser.add_argument("--dataset-id", type=str, default="Video-R1/Video-R1-data")
    parser.add_argument("--manifest-name", type=str, default="Video-R1-COT-165k.json")
    parser.add_argument("--dataset-dir", type=str, default="data/video_r1_cot/raw")
    parser.add_argument("--processed-dir", type=str, default="data/video_r1_cot/processed")
    parser.add_argument("--processed-name", type=str, default="train.jsonl")
    parser.add_argument("--summary-name", type=str, default="video_r1_cot_summary.json")
    parser.add_argument("--sample-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subsets", type=str, default="")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--max-frame-size", type=int, default=768)
    parser.add_argument(
        "--download-mode",
        type=str,
        default="subset-directories",
        choices=("sampled-files", "subset-directories"),
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-archive-extract", action="store_true")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    processed_dir = Path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    selected_subsets = [item.strip() for item in args.subsets.split(",") if item.strip()]

    download_dataset_files(
        dataset_id=args.dataset_id,
        local_dir=dataset_dir,
        allow_patterns=[args.manifest_name],
        token=hf_token_from_env(),
    )

    manifest_path = dataset_dir / args.manifest_name
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    dataset = load_dataset("json", data_files=str(manifest_path), split="train")

    filtered_rows: list[dict] = []
    per_subset_counts = Counter()
    skipped_non_video = 0

    for row in dataset:
        subset = match_subset(row, selected_subsets or None)
        if not subset:
            continue
        if not is_video_row(row):
            skipped_non_video += 1
            continue

        path_value = normalize_repo_path(str(row.get("path") or ""))
        question = pick_question(row)
        gold_answer = normalize_answer_text(pick_gold_answer(row))
        options = pick_options(row)

        row_copy = {
            "question": question,
            "options": options,
            "gold_answer": gold_answer,
            "video_path": path_value,
            "source_subset": subset,
        }
        filtered_rows.append(row_copy)
        per_subset_counts[subset] += 1

    sampled_rows, sampling_stats = sample_rows_by_subset(
        rows=filtered_rows,
        sample_ratio=args.sample_ratio,
        seed=args.seed,
    )

    download_stats = {"requested_video_files": 0}
    if not args.skip_download:
        if args.download_mode == "sampled-files":
            requested_paths = sorted({row["video_path"] for row in sampled_rows if row.get("video_path")})
            download_dataset_files(
                dataset_id=args.dataset_id,
                local_dir=dataset_dir,
                allow_patterns=[args.manifest_name, *requested_paths],
                token=hf_token_from_env(),
            )
            download_stats = {"requested_video_files": len(requested_paths)}
        else:
            subset_dirs = sorted({str(row.get("source_subset") or "") for row in sampled_rows if str(row.get("source_subset") or "")})
            download_dataset_files(
                dataset_id=args.dataset_id,
                local_dir=dataset_dir,
                allow_patterns=[args.manifest_name, *(f"{subset}/**" for subset in subset_dirs)],
                token=hf_token_from_env(),
            )

    archive_stats = {"zip_archives": 0, "tar_archives": 0, "archives_extracted": 0}
    if not args.skip_archive_extract:
        archive_stats = extract_archives(dataset_dir)

    processed_rows, frame_stats = extract_frames_for_rows(
        rows=sampled_rows,
        split_name="train",
        video_root=dataset_dir,
        frame_root=processed_dir / "frames",
        num_frames=args.num_frames,
        max_size=args.max_frame_size,
        progress_label="video_r1_cot",
    )

    processed_path = processed_dir / args.processed_name
    write_jsonl(processed_path, processed_rows)

    summary = {
        "dataset_id": args.dataset_id,
        "manifest": str(manifest_path.resolve()),
        "selected_subsets": selected_subsets,
        "filtered_rows": len(filtered_rows),
        "sampled_rows": len(sampled_rows),
        "sample_ratio": args.sample_ratio,
        "sampling_seed": args.seed,
        "sampling_stats": sampling_stats,
        "download_mode": args.download_mode,
        "download_stats": download_stats,
        "archive_stats": archive_stats,
        "per_subset_counts": dict(per_subset_counts),
        "skipped_non_video": skipped_non_video,
        "processed_path": str(processed_path.resolve()),
        "frame_stats": frame_stats,
    }
    write_json(processed_dir / args.summary_name, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
