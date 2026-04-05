import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


FRAME_GLOB_PATTERNS = ("frame_*.jpg", "frame_*.jpeg", "frame_*.png", "*.jpg", "*.jpeg", "*.png")
SUPPORTED_MODES = {"length", "perspective"}
ALLOWED_REASONING_TYPES = {"ABSTRACT", "TEMPORAL", "SPATIOTEMPORAL"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert raw annotation JSONL into SFT-ready JSONL.")
    parser.add_argument("--mode", choices=sorted(SUPPORTED_MODES), required=True)
    parser.add_argument("--input", type=str, required=True, help="Raw annotation JSON/JSONL path.")
    parser.add_argument("--output", type=str, required=True, help="Output SFT JSONL path.")
    parser.add_argument(
        "--frames-root",
        type=str,
        default=None,
        help="Root directory containing train/test frame subdirs. Defaults to <input_dir>/frames.",
    )
    parser.add_argument("--frames-per-sample", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--summary", type=str, default=None, help="Optional summary JSON path.")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        if path.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of rows in {path}")
    return data


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def normalize_answer_text(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    lower_raw = raw.lower()
    start_tag = "<answer>"
    end_tag = "</answer>"
    if start_tag in lower_raw and end_tag in lower_raw:
        start_idx = lower_raw.index(start_tag) + len(start_tag)
        end_idx = lower_raw.index(end_tag, start_idx)
        return raw[start_idx:end_idx].strip()
    return raw


def build_question_with_options(question: str, options: list[str]) -> str:
    question_text = str(question or "").strip()
    option_block = "\n".join(str(opt).strip() for opt in options if str(opt).strip())
    if not option_block:
        return question_text
    return f"{question_text}\n\nOptions:\n{option_block}"


def collect_frame_paths_from_subdir(frame_subdir: str, frames_root: Path, frames_per_sample: int) -> list[Path]:
    normalized_subdir = str(frame_subdir or "").strip()
    if not normalized_subdir:
        return []
    for split_name in ("train", "test"):
        frame_dir = frames_root / split_name / normalized_subdir
        if frame_dir.exists():
            frames: list[Path] = []
            for pattern in FRAME_GLOB_PATTERNS:
                frames.extend(frame_dir.glob(pattern))
            return sorted({path.resolve() for path in frames if path.is_file()})[:frames_per_sample]
    return []


def relativize_paths(paths: list[Path], output_dir: Path) -> list[str]:
    serialized: list[str] = []
    for path in paths:
        try:
            serialized.append(os.path.relpath(path, output_dir))
        except ValueError:
            serialized.append(str(path))
    return serialized


def resolve_media(
    row: dict[str, Any],
    input_path: Path,
    output_dir: Path,
    frames_root: Path,
    frames_per_sample: int,
) -> dict[str, Any]:
    explicit_frames = row.get("frames")
    if isinstance(explicit_frames, list):
        resolved_frames: list[Path] = []
        for item in explicit_frames:
            text = str(item or "").strip()
            if not text:
                continue
            path = Path(text)
            if not path.is_absolute():
                path = (input_path.parent / path).resolve()
            if path.exists():
                resolved_frames.append(path)
        if resolved_frames:
            return {"frames": relativize_paths(resolved_frames[:frames_per_sample], output_dir)}

    frame_subdir = str(row.get("frame_subdir") or "").strip()
    if frame_subdir:
        frames = collect_frame_paths_from_subdir(frame_subdir, frames_root, frames_per_sample)
        if frames:
            return {"frames": relativize_paths(frames, output_dir)}

    for key in ("image", "image_path", "video_path"):
        text = str(row.get(key) or "").strip()
        if not text:
            continue
        path = Path(text)
        if not path.is_absolute():
            path = (input_path.parent / text).resolve()
        if path.exists() and path.is_file():
            return {"image": relativize_paths([path], output_dir)[0]}

    return {}


def export_length_rows(
    rows: list[dict[str, Any]],
    input_path: Path,
    output_path: Path,
    frames_root: Path,
    frames_per_sample: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    exported: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()

    for row in rows:
        question = str(row.get("question") or "")
        options = [str(opt) for opt in (row.get("options") or [])]
        instruction = build_question_with_options(question=question, options=options)
        answer_raw = str(row.get("answer_raw") or "").strip()
        cot_raw = str(row.get("cot_raw") or "").strip()
        long_cot_raw = str(row.get("long_cot_raw") or "").strip()
        if not instruction:
            stats["skip_missing_instruction"] += 1
            continue
        if not answer_raw:
            stats["skip_missing_answer"] += 1
            continue

        media_fields = resolve_media(row, input_path, output_path.parent, frames_root, frames_per_sample)
        if not media_fields:
            stats["skip_missing_media"] += 1
            continue

        base = {
            "instruction": instruction,
            "input": "",
            "video_path": row.get("video_path"),
            "source_subset": row.get("source_subset"),
            "frame_subdir": row.get("frame_subdir"),
            "gold_answer": normalize_answer_text(str(row.get("gold_answer") or answer_raw)),
            "sft_mode": "length",
            **media_fields,
        }

        exported.append(
            {
                **base,
                "output": f"<ANSWER>\n{answer_raw}\n</ANSWER>",
                "reasoning_depth": "ANSWER",
            }
        )
        stats["answer_rows"] += 1

        if cot_raw:
            exported.append(
                {
                    **base,
                    "output": f"<COT>\n{cot_raw}\n</COT>\n<ANSWER>\n{answer_raw}\n</ANSWER>",
                    "reasoning_depth": "COT",
                }
            )
            stats["cot_rows"] += 1
        else:
            stats["skip_missing_cot"] += 1

        if long_cot_raw:
            exported.append(
                {
                    **base,
                    "output": f"<LONG_COT>\n{long_cot_raw}\n</LONG_COT>\n<ANSWER>\n{answer_raw}\n</ANSWER>",
                    "reasoning_depth": "LONG_COT",
                }
            )
            stats["long_cot_rows"] += 1
        else:
            stats["skip_missing_long_cot"] += 1

    return exported, dict(stats)


def export_perspective_rows(
    rows: list[dict[str, Any]],
    input_path: Path,
    output_path: Path,
    frames_root: Path,
    frames_per_sample: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    exported: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()

    for row in rows:
        question = str(row.get("question") or "")
        options = [str(opt) for opt in (row.get("options") or [])]
        instruction = build_question_with_options(question=question, options=options)
        reasoning_type = str(row.get("granularity_type") or row.get("reasoning_type") or "").strip().upper()
        reasoning_trace = str(
            row.get("granularity_thinking_raw")
            or row.get("reasoning_raw")
            or row.get("thinking")
            or ""
        ).strip()
        answer_text = normalize_answer_text(str(row.get("gold_answer") or row.get("answer") or ""))

        if not instruction:
            stats["skip_missing_instruction"] += 1
            continue
        if not reasoning_type:
            stats["skip_missing_reasoning_type"] += 1
            continue
        if reasoning_type not in ALLOWED_REASONING_TYPES:
            stats["skip_invalid_reasoning_type"] += 1
            continue
        if not reasoning_trace:
            stats["skip_missing_reasoning_trace"] += 1
            continue
        if not answer_text:
            stats["skip_missing_answer"] += 1
            continue

        media_fields = resolve_media(row, input_path, output_path.parent, frames_root, frames_per_sample)
        if not media_fields:
            stats["skip_missing_media"] += 1
            continue

        exported.append(
            {
                "instruction": instruction,
                "input": "",
                "output": (
                    f"<REASONING_TYPE>\n{reasoning_type}\n</REASONING_TYPE>\n"
                    f"<REASONING>\n{reasoning_trace}\n</REASONING>\n"
                    f"<ANSWER>\n{answer_text}\n</ANSWER>"
                ),
                "reasoning_type": reasoning_type,
                "video_path": row.get("video_path"),
                "source_subset": row.get("source_subset"),
                "frame_subdir": row.get("frame_subdir"),
                "gold_answer": answer_text,
                "sft_mode": "perspective",
                **media_fields,
            }
        )
        stats["perspective_rows"] += 1
        stats[f"reasoning_type:{reasoning_type}"] += 1

    return exported, dict(stats)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    frames_root = Path(args.frames_root).resolve() if args.frames_root else (input_path.parent / "frames").resolve()

    rows = load_rows(input_path)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    if args.mode == "length":
        exported_rows, stats = export_length_rows(rows, input_path, output_path, frames_root, args.frames_per_sample)
    else:
        exported_rows, stats = export_perspective_rows(rows, input_path, output_path, frames_root, args.frames_per_sample)

    write_jsonl(output_path, exported_rows)

    summary = {
        "mode": args.mode,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "frames_root": str(frames_root),
        "input_rows": len(rows),
        "exported_rows": len(exported_rows),
        "stats": stats,
    }
    if args.summary:
        write_json(Path(args.summary).resolve(), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
