import json
import os
import random
import re
import tarfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

import av
import numpy as np
from huggingface_hub import snapshot_download
from PIL import Image


FRAME_GLOB_PATTERNS = ("frame_*.jpg", "frame_*.jpeg", "frame_*.png", "*.jpg", "*.jpeg", "*.png")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_stem(text: str) -> str:
    keep: list[str] = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "sample"


def frame_key_from_path(path_text: str) -> str:
    path = Path(path_text)
    parts = [safe_stem(part) for part in path.with_suffix("").parts if part not in ("", ".", "..")]
    return "__".join(part for part in parts if part) or safe_stem(path.stem)


def hf_token_from_env() -> Optional[str]:
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")


def download_dataset_files(
    dataset_id: str,
    local_dir: Path,
    allow_patterns: list[str],
    token: Optional[str] = None,
) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=dataset_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        allow_patterns=allow_patterns,
        token=token,
        resume_download=True,
    )


def extract_archives(dataset_root: Path) -> dict[str, int]:
    stats = {"zip_archives": 0, "tar_archives": 0, "archives_extracted": 0}
    for archive in dataset_root.rglob("*"):
        if not archive.is_file():
            continue

        suffixes = archive.suffixes
        sentinel = archive.with_name(f".{archive.name}.extracted")
        if sentinel.exists():
            continue

        if archive.suffix.lower() == ".zip":
            stats["zip_archives"] += 1
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(archive.parent)
            sentinel.write_text("", encoding="utf-8")
            stats["archives_extracted"] += 1
            continue

        if suffixes[-2:] == [".tar", ".gz"] or archive.suffix.lower() in {".tar", ".tgz"}:
            stats["tar_archives"] += 1
            with tarfile.open(archive) as tf:
                tf.extractall(archive.parent)
            sentinel.write_text("", encoding="utf-8")
            stats["archives_extracted"] += 1
    return stats


def build_video_index(video_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in video_root.rglob("*.mp4"):
        index.setdefault(path.name, path)
    return index


def resolve_video_path(
    sample: dict,
    video_root: Path,
    index: dict[str, Path],
    keys: tuple[str, ...] = ("video_id", "video", "video_path", "video_name", "path"),
) -> Optional[Path]:
    for key in keys:
        value = sample.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        candidate = Path(value)
        if candidate.exists():
            return candidate.resolve()
        joined = video_root / value
        if joined.exists():
            return joined.resolve()
        stripped = video_root / value.lstrip("./")
        if stripped.exists():
            return stripped.resolve()
        by_name = index.get(Path(value).name)
        if by_name is not None and by_name.exists():
            return by_name.resolve()
    return None


def load_video_frames(video_path: Path, num_frames: int, max_size: int) -> list[Image.Image]:
    try:
        container = av.open(str(video_path))
        stream = container.streams.video[0]

        total_frames = stream.frames
        if not total_frames:
            total_frames = sum(1 for _ in container.decode(video=0))
            container.seek(0)

        indices = np.linspace(0, max(total_frames - 1, 0), num_frames, dtype=int)
        idx_set = set(indices.tolist())

        frames: list[Image.Image] = []
        for idx, frame in enumerate(container.decode(video=0)):
            if idx in idx_set:
                image = frame.to_image()
                if max(image.size) > max_size:
                    ratio = max_size / float(max(image.size))
                    resized = (max(1, int(image.size[0] * ratio)), max(1, int(image.size[1] * ratio)))
                    image = image.resize(resized, Image.Resampling.LANCZOS)
                frames.append(image)
            if len(frames) >= num_frames:
                break

        container.close()
        while len(frames) < num_frames:
            frames.append(frames[-1] if frames else Image.new("RGB", (224, 224)))
        return frames[:num_frames]
    except Exception:
        return [Image.new("RGB", (224, 224)) for _ in range(num_frames)]


def extract_frames_for_rows(
    rows: list[dict],
    split_name: str,
    video_root: Path,
    frame_root: Path,
    num_frames: int,
    max_size: int,
    progress_label: str,
) -> tuple[list[dict], dict[str, int]]:
    frame_root.mkdir(parents=True, exist_ok=True)
    video_index = build_video_index(video_root)
    cached_by_video: dict[Path, str] = {}

    processed_rows: list[dict] = []
    missing_videos = 0
    decoded_videos = 0

    for idx, row in enumerate(rows):
        video_path = resolve_video_path(row, video_root, video_index)
        if video_path is None:
            missing_videos += 1
            continue

        resolved_video = video_path.resolve()
        frame_subdir = cached_by_video.get(resolved_video)
        if frame_subdir is None:
            subset = str(row.get("source_subset") or row.get("question_category") or "unknown")
            frame_subdir = str(Path(subset) / frame_key_from_path(str(resolved_video.relative_to(video_root.resolve()))))
            output_dir = frame_root / split_name / frame_subdir
            output_dir.mkdir(parents=True, exist_ok=True)

            existing_frames = sorted(output_dir.glob("frame_*.jpg"))
            if len(existing_frames) < num_frames:
                frames = load_video_frames(video_path=resolved_video, num_frames=num_frames, max_size=max_size)
                for frame_idx, frame in enumerate(frames):
                    frame.save(output_dir / f"frame_{frame_idx:03d}.jpg", format="JPEG", quality=95)

            cached_by_video[resolved_video] = frame_subdir
            decoded_videos += 1
            if decoded_videos % 50 == 0:
                print(f"[{progress_label}] decoded videos: {decoded_videos} (row index: {idx})")

        row_copy = dict(row)
        row_copy["frame_subdir"] = frame_subdir
        processed_rows.append(row_copy)

    stats = {
        "input_rows": len(rows),
        "processed_rows": len(processed_rows),
        "decoded_videos": decoded_videos,
        "missing_videos": missing_videos,
    }
    return processed_rows, stats


def collect_frame_paths_from_subdir(frame_subdir: str, frames_root: Path, frames_per_sample: int) -> list[Path]:
    frame_subdir = str(frame_subdir or "").strip()
    if not frame_subdir:
        return []
    frame_dir = frames_root / "train" / frame_subdir
    if not frame_dir.exists():
        frame_dir = frames_root / "test" / frame_subdir
    if not frame_dir.exists():
        return []
    frames: list[Path] = []
    for pattern in FRAME_GLOB_PATTERNS:
        frames.extend(frame_dir.glob(pattern))
    unique_frames = sorted({path.resolve() for path in frames if path.is_file()})
    return unique_frames[:frames_per_sample]


def collect_frame_paths(processed_row: dict, processed_jsonl_path: Path, frames_per_sample: int) -> list[Path]:
    frames_root = processed_jsonl_path.parent / "frames"
    frame_subdir = str(processed_row.get("frame_subdir") or "").strip()
    return collect_frame_paths_from_subdir(frame_subdir, frames_root, frames_per_sample)


def pick_question(row: dict) -> str:
    for key in ("question", "problem", "prompt"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    conversations = row.get("conversations")
    if isinstance(conversations, list):
        for item in conversations:
            if isinstance(item, dict):
                role = str(item.get("from") or item.get("role") or "").lower()
                value = str(item.get("value") or item.get("content") or "").strip()
                if role in {"human", "user"} and value:
                    return value
    return ""


def pick_gold_answer(row: dict) -> str:
    for key in ("answer", "solution", "output", "response"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    conversations = row.get("conversations")
    if isinstance(conversations, list):
        for item in conversations:
            if isinstance(item, dict):
                role = str(item.get("from") or item.get("role") or "").lower()
                value = str(item.get("value") or item.get("content") or "").strip()
                if role in {"gpt", "assistant"} and value:
                    return value
    return ""


def pick_options(row: dict) -> list[str]:
    options = row.get("options") or row.get("choices") or row.get("choice")
    if isinstance(options, list):
        return [str(opt).strip() for opt in options if str(opt).strip()]
    if isinstance(options, dict):
        normalized: list[str] = []
        for key in sorted(options.keys()):
            value = str(options[key] or "").strip()
            if value:
                normalized.append(f"{key}. {value}")
        return normalized
    return []


def is_video_row(row: dict) -> bool:
    return str(row.get("data_type") or "").strip().lower() == "video" or bool(str(row.get("path") or "").strip())


def sample_rows_by_subset(rows: list[dict], sample_ratio: float, seed: int) -> tuple[list[dict], dict[str, dict[str, int]]]:
    if not 0 < sample_ratio <= 1.0:
        raise ValueError(f"sample_ratio must be in (0, 1], got {sample_ratio}")
    if sample_ratio >= 1.0:
        counts = Counter(str(row.get("source_subset") or "unknown") for row in rows)
        return rows, {subset: {"total": total, "sampled": total} for subset, total in counts.items()}

    rng = random.Random(seed)
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        subset = str(row.get("source_subset") or "unknown")
        grouped.setdefault(subset, []).append(row)

    sampled_rows: list[dict] = []
    stats: dict[str, dict[str, int]] = {}
    for subset in sorted(grouped.keys()):
        subset_rows = grouped[subset]
        total = len(subset_rows)
        sample_n = min(total, max(1, round(total * sample_ratio)))
        sampled = rng.sample(subset_rows, sample_n)
        sampled_rows.extend(sampled)
        stats[subset] = {"total": total, "sampled": sample_n}
    rng.shuffle(sampled_rows)
    return sampled_rows, stats


JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
TAG_BLOCK_RE = re.compile(r"<\s*([A-Za-z0-9_]+)\s*>(.*?)<\s*/\s*\1\s*>", re.DOTALL)


def parse_json_block(text: str) -> Optional[dict]:
    match = JSON_BLOCK_RE.search(text)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def extract_xml_tag_text(text: str, tag: str) -> str:
    pattern = re.compile(rf"<\s*{re.escape(tag)}\s*>(.*?)<\s*/\s*{re.escape(tag)}\s*>", re.DOTALL | re.IGNORECASE)
    match = pattern.search(str(text or ""))
    if not match:
        return ""
    return match.group(1).strip()


def strip_outer_reasoning_tag(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    match = TAG_BLOCK_RE.fullmatch(raw)
    if match:
        return match.group(2).strip()
    return raw


def normalize_answer_text(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    tagged = extract_xml_tag_text(raw, "answer")
    return tagged or raw


def build_question_with_options(question: str, options: list[str]) -> str:
    question_text = str(question or "").strip()
    if not options:
        return question_text
    option_block = "\n".join(str(opt).strip() for opt in options if str(opt).strip())
    if not option_block:
        return question_text
    return f"{question_text}\n\nOptions:\n{option_block}"
