import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml
from PIL import Image
from PIL import UnidentifiedImageError
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from .prompts import (
    ANSWER_STYLE_LABELS,
    REASONING_TYPE_LABELS,
    build_annotation_prompt,
    build_generation_prompt,
    build_granularity_generation_prompt,
    generation_system_prompt,
    granularity_generation_system_prompt,
    system_prompt,
)
from .utils import collect_frame_paths, load_jsonl, normalize_answer_text, parse_json_block, write_json, write_jsonl


@dataclass
class AnnotateConfig:
    model_name_or_path: str
    input_path: str
    output_dir: str
    output_name: str
    summary_name: str
    annotation_task: str
    max_samples: Optional[int] = None
    frames_per_sample: int = 8
    temperature: float = 0.0
    max_completion_tokens: int = 256
    gpu_memory_utilization: float = 0.85
    device: str = "cuda"
    max_model_len: int = 8192
    max_pixels: int = 501760
    min_pixels: int = 3136


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate processed Video-R1-COT samples with Qwen2.5-VL-72B.")
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def load_config(path: str) -> AnnotateConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return AnnotateConfig(**raw)


def resize_image_to_pixel_bounds(image: Image.Image, max_pixels: int, min_pixels: int) -> Image.Image:
    width, height = image.size
    if width <= 0 or height <= 0:
        return image
    pixels = width * height
    target_pixels = pixels
    if pixels > max_pixels:
        target_pixels = max_pixels
    elif pixels < min_pixels:
        target_pixels = min_pixels
    if target_pixels == pixels:
        return image
    scale = (target_pixels / float(pixels)) ** 0.5
    new_w = max(1, int(width * scale))
    new_h = max(1, int(height * scale))
    return image.resize((new_w, new_h), Image.Resampling.BICUBIC)


GENERATION_TASK = "generate_sft"
GRANULARITY_GENERATION_TASK = "generate_granularity"


def allowed_labels(task: str) -> set[str]:
    if task == "answer_style":
        return set(ANSWER_STYLE_LABELS)
    if task == "reasoning_type":
        return set(REASONING_TYPE_LABELS)
    raise ValueError(f"Unsupported annotation task: {task}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    input_path = Path(cfg.input_path)
    rows = load_jsonl(input_path)
    if cfg.max_samples is not None:
        rows = rows[: cfg.max_samples]
    if not rows:
        raise SystemExit(f"No rows found in {input_path}")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(cfg.model_name_or_path, trust_remote_code=False)
    llm = LLM(
        model=cfg.model_name_or_path,
        tokenizer=cfg.model_name_or_path,
        dtype="bfloat16",
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        max_model_len=cfg.max_model_len,
        enforce_eager=True,
        limit_mm_per_prompt={"image": cfg.frames_per_sample},
    )
    sampling_params = SamplingParams(
        temperature=cfg.temperature,
        max_tokens=cfg.max_completion_tokens,
        n=1,
    )

    is_generation = cfg.annotation_task == GENERATION_TASK
    is_granularity_generation = cfg.annotation_task == GRANULARITY_GENERATION_TASK
    labels = None if (is_generation or is_granularity_generation) else allowed_labels(cfg.annotation_task)
    label_counts = Counter()
    annotated_rows: list[dict] = []
    skipped_invalid_frames = 0

    for idx, row in enumerate(rows):
        frame_paths = collect_frame_paths(
            processed_row=row,
            processed_jsonl_path=input_path,
            frames_per_sample=cfg.frames_per_sample,
        )
        if not frame_paths:
            continue

        try:
            frames = [
                resize_image_to_pixel_bounds(Image.open(path).convert("RGB"), cfg.max_pixels, cfg.min_pixels)
                for path in frame_paths
            ]
        except (UnidentifiedImageError, OSError):
            skipped_invalid_frames += 1
            continue

        question = str(row.get("question") or "")
        options = [str(opt) for opt in row.get("options") or []]
        gold_answer = normalize_answer_text(str(row.get("gold_answer") or row.get("answer") or ""))

        if is_generation:
            user_prompt = build_generation_prompt(question=question, options=options, gold_answer=gold_answer)
            sys_prompt_text = generation_system_prompt()
        elif is_granularity_generation:
            user_prompt = build_granularity_generation_prompt(
                question=question,
                options=options,
                gold_answer=gold_answer,
            )
            sys_prompt_text = granularity_generation_system_prompt()
        else:
            user_prompt = build_annotation_prompt(
                task=cfg.annotation_task,
                question=question,
                options=options,
                gold_answer=gold_answer,
            )
            sys_prompt_text = system_prompt(cfg.annotation_task)

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": sys_prompt_text}],
            },
            {
                "role": "user",
                "content": ([{"type": "image"} for _ in frames] + [{"type": "text", "text": user_prompt}]),
            },
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = llm.generate(
            [{"prompt": prompt, "multi_modal_data": {"image": frames}}],
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        text = outputs[0].outputs[0].text
        payload = parse_json_block(text) or {}

        base_row = {
            "question": row.get("question"),
            "options": row.get("options"),
            "gold_answer": gold_answer,
            "video_path": row.get("video_path"),
            "source_subset": row.get("source_subset"),
            "frame_subdir": row.get("frame_subdir"),
        }

        if is_generation:
            answer_raw = str(payload.get("answer") or "").strip()
            cot_raw = str(payload.get("cot") or "").strip()
            long_cot_raw = str(payload.get("long_cot") or "").strip()
            has_all = bool(answer_raw and cot_raw and long_cot_raw)
            label_counts["complete" if has_all else "incomplete"] += 1
            annotated_rows.append({
                **base_row,
                "answer_raw": answer_raw,
                "cot_raw": cot_raw,
                "long_cot_raw": long_cot_raw,
            })
        elif is_granularity_generation:
            granularity_type = str(payload.get("granularity_type") or "").strip().upper()
            thinking = str(payload.get("thinking") or "").strip()
            if granularity_type not in set(REASONING_TYPE_LABELS):
                granularity_type = "INVALID"
            label_counts[granularity_type] += 1
            if not thinking:
                label_counts["missing_thinking"] += 1
            annotated_rows.append({
                **base_row,
                "annotation_task": cfg.annotation_task,
                "granularity_type": granularity_type,
                "granularity_thinking_raw": thinking,
                "model_response": text,
            })
        else:
            label = str(payload.get("label") or "").strip().upper()
            reason = str(payload.get("reason") or "").strip()
            if label not in labels:  # type: ignore[operator]
                label = "INVALID"
            label_counts[label] += 1
            annotated_rows.append({
                **base_row,
                "annotation_task": cfg.annotation_task,
                "frames": [str(path) for path in frame_paths],
                "model_response": text,
                "annotation_label": label,
                "annotation_reason": reason,
            })

        if (idx + 1) % 20 == 0:
            print(f"[annotate] processed {idx + 1} rows")

    output_path = output_dir / cfg.output_name
    write_jsonl(output_path, annotated_rows)

    summary = {
        "model_name_or_path": cfg.model_name_or_path,
        "annotation_task": cfg.annotation_task,
        "input_path": str(input_path.resolve()),
        "output_path": str(output_path.resolve()),
        "input_rows": len(rows),
        "annotated_rows": len(annotated_rows),
        "skipped_invalid_frames": skipped_invalid_frames,
        "label_counts": dict(label_counts),
    }
    write_json(output_dir / cfg.summary_name, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
