from typing import Any


ANSWER_STYLE_LABELS = ("DIRECT_ANSWER", "COT", "LONG_COT")
REASONING_TYPE_LABELS = ("ABSTRACT", "TEMPORAL", "SPATIOTEMPORAL")
REASONING_DEPTH_LABELS = ("ANSWER", "COT", "LONG_COT")

'''
여기서 두 가지 Style의 Annotation Prompt를 정의합니다.

- Answer Style: 답변의 형태를 정의합니다.
- Reasoning Type: 답변의 이유를 정의합니다.

Answer Style:
- DIRECT_ANSWER: 답변은 직접 관찰 또는 단순한 단계로 주어집니다.
- COT: 답변은 짧은 여러 단계의 추론으로 주어집니다.
- LONG_COT: 답변은 긴 여러 단계의 추론으로 주어집니다.

Reasoning Type:
- ABSTRACT: 추상적인 이유를 정의합니다.
- TEMPORAL: 시간적인 이유를 정의합니다.
- SPATIOTEMPORAL: 공간적인 이유를 정의합니다.
'''

def build_annotation_prompt(task: str, question: str, options: list[str], gold_answer: str) -> str:
    option_block = "\n".join(options) if options else ""

    if task == "answer_style":
        return (
            "You are annotating the reasoning style required to answer a video question.\n\n"
            "Choose exactly one label:\n"
            "- DIRECT_ANSWER: the answer can be given from direct observation or a single trivial step.\n"
            "- COT: the answer requires short multi-step reasoning over the video.\n"
            "- LONG_COT: the answer requires extended reasoning, synthesis across multiple cues, or deeper chain-of-thought.\n\n"
            "Return strict JSON with keys `label` and `reason`.\n"
            f"Question:\n{question}\n\n"
            f"Options:\n{option_block}\n\n"
            f"Gold answer:\n{gold_answer}\n"
        )

    if task == "reasoning_type":
        return (
            "You are annotating the dominant reasoning type required by a video question.\n\n"
            "Choose exactly one label:\n"
            "- ABSTRACT: semantics, intent, category, commonsense, or non-temporal reasoning is primary.\n"
            "- TEMPORAL: ordering, duration, before/after, progression, or time-dependent reasoning is primary.\n"
            "- SPATIOTEMPORAL: both motion/spatial relations and temporal evolution are central.\n\n"
            "Return strict JSON with keys `label` and `reason`.\n"
            f"Question:\n{question}\n\n"
            f"Options:\n{option_block}\n\n"
            f"Gold answer:\n{gold_answer}\n"
        )

    raise ValueError(f"Unsupported annotation task: {task}")


def build_generation_prompt(question: str, options: list[str], gold_answer: str) -> str:
    option_block = "\n".join(options) if options else ""
    return (
        "You are constructing training data for a multimodal reasoning model.\n"
        "The input includes video frames, a question, answer options, and the correct answer.\n\n"
        "Your task is to generate THREE outputs with different reasoning depths based on the video content.\n\n"
        "Outputs:\n"
        "1) answer: Direct answer only, no reasoning.\n"
        "2) cot: Minimal reasoning using only necessary visual and/or temporal cues from the frames.\n"
        "3) long_cot: Detailed reasoning that integrates multiple visual cues, step decomposition, or temporal understanding across frames.\n\n"
        "Constraints:\n"
        "- All outputs must produce the SAME final answer.\n"
        "- Reasoning must be grounded in the video frames — cite specific visual evidence.\n"
        "- Do not include conversational or stylistic phrases.\n"
        "- Do not use speculative or unsupported assumptions.\n"
        "- Avoid redundancy between cot and long_cot.\n\n"
        "Depth definition:\n"
        "- cot: Include only the essential visual cues from the frames needed to reach the answer.\n"
        "- long_cot: Must include at least one of the following:\n"
        "  * temporal reasoning across multiple frames (ordering, progression, before/after)\n"
        "  * integration of multiple distinct visual cues from different frames\n"
        "  * decomposition into intermediate steps with explicit frame references\n"
        "  * explicit consistency verification across the visual sequence\n\n"
        "Strict rules:\n"
        "- long_cot must NOT be a paraphrase or simple extension of cot.\n"
        "- Each step in long_cot must add new reasoning value beyond what cot covers.\n"
        "- The final answer in all three outputs must exactly match the provided correct answer.\n\n"
        "Output format (STRICT JSON):\n"
        "{\n"
        '  "answer": "<direct answer only, no explanation>",\n'
        '  "cot": "<concise step-by-step reasoning grounded in key frame observations>",\n'
        '  "long_cot": "<detailed multi-step reasoning with explicit visual and temporal grounding>"\n'
        "}\n\n"
        "Input:\n"
        "- Video frames: provided as visual sequence\n"
        f"- Question:\n{question}\n\n"
        f"- Options:\n{option_block}\n\n"
        f"- Correct answer:\n{gold_answer}\n"
    )


def build_granularity_generation_prompt(question: str, options: list[str], gold_answer: str) -> str:
    option_block = "\n".join(options) if options else ""
    return (
        "You are constructing training data for a multimodal video reasoning model.\n"
        "The input includes video frames, a question, answer options, and the correct answer.\n\n"
        "Your task is to choose the SINGLE most appropriate granularity type for solving the question, "
        "then write one reasoning annotation grounded in the frames from that perspective.\n\n"
        "Granularity types:\n"
        "- ABSTRACT: semantic or meaning-level understanding is primary. This includes category recognition, "
        "caption-like understanding, intent, high-level VQA, and retrieval-style semantics.\n"
        "- TEMPORAL: time and sequence understanding is primary. This includes ordering, progression, "
        "before/after relations, temporal localization, temporal grounding, and summarization over time.\n"
        "- SPATIOTEMPORAL: both spatial relations and temporal evolution are primary. This includes motion, "
        "object interaction over time, tracking, and spatiotemporal grounding.\n\n"
        "Instructions:\n"
        "- First decide which granularity type best matches the reasoning required by the question.\n"
        "- Then write a concise but meaningful thinking trace from that granularity perspective.\n"
        "- The thinking must be grounded in the frames and must support the provided correct answer.\n"
        "- Do not write three candidates. Choose exactly one granularity type.\n"
        "- Do not output unsupported assumptions.\n"
        "- Do not include conversational filler.\n\n"
        "Output format (STRICT JSON):\n"
        "{\n"
        '  "granularity_type": "ABSTRACT or TEMPORAL or SPATIOTEMPORAL",\n'
        '  "thinking": "<reasoning annotation written from the selected granularity perspective>"\n'
        "}\n\n"
        "Input:\n"
        "- Video frames: provided as visual sequence\n"
        f"- Question:\n{question}\n\n"
        f"- Options:\n{option_block}\n\n"
        f"- Correct answer:\n{gold_answer}\n"
    )


def generation_system_prompt() -> str:
    return (
        "You are a careful video-annotation assistant.\n"
        "Analyze the provided video frames, question, options, and correct answer.\n"
        "Generate three reasoning outputs (answer, cot, long_cot) of increasing depth, "
        "all grounded in what you observe in the frames.\n"
        "Respond with valid JSON only. Example: "
        '{"answer": "...", "cot": "...", "long_cot": "..."}'
    )


def granularity_generation_system_prompt() -> str:
    return (
        "You are a careful video-annotation assistant.\n"
        "Analyze the provided video frames, question, options, and correct answer.\n"
        "Choose the single best granularity type among ABSTRACT, TEMPORAL, and SPATIOTEMPORAL, "
        "then generate one reasoning annotation from that perspective.\n"
        "Respond with valid JSON only. Example: "
        '{"granularity_type": "TEMPORAL", "thinking": "..."}'
    )


def system_prompt(task: str) -> str:
    if task == "answer_style":
        labels = ", ".join(ANSWER_STYLE_LABELS)
    elif task == "reasoning_type":
        labels = ", ".join(REASONING_TYPE_LABELS)
    else:
        raise ValueError(f"Unsupported annotation task: {task}")

    return (
        "You are a careful video-annotation assistant.\n"
        "Look at the provided video frames, question, options, and gold answer.\n"
        f"Choose exactly one label from: {labels}.\n"
        "Respond with valid JSON only. Example: {\"label\": \"...\", \"reason\": \"...\"}"
    )
