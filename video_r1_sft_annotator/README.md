# Video-R1 SFT Annotator

`Video-R1-COT-165k`를 기반으로, `Qwen/Qwen2.5-VL-72B-Instruct`를 사용해 비디오 질문 하나당 세 가지 추론 깊이의 응답(`ANSWER`, `COT`, `LONG_COT`)을 동시에 생성하고, 최종적으로 SFT 파이프라인이 바로 읽을 수 있는 `instruction / input / output` 형식의 JSON을 만드는 전용 프레임워크다.

추가로, 각 문제에 대해 `ABSTRACT / TEMPORAL / SPATIOTEMPORAL` 중 하나를 선택하고 그 관점의 reasoning을 생성하는 perspective SFT 데이터도 export할 수 있다.

## 핵심 목적

비디오 질문 하나에 대해 세 가지 추론 형태를 한 번의 모델 호출로 생성한다.

- `ANSWER`: 추론 없이 정답만
- `COT`: 핵심 시각 단서만 사용한 간결한 추론
- `LONG_COT`: 여러 프레임을 아우르는 시간적·공간적 추론을 포함한 심층 추론

입력 1개 → 훈련 샘플 3개로 확장된다.

## 1. 전체 흐름

```text
Video-R1-COT-165k manifest 다운로드
→ video row만 필터링
→ subset 선택 및 샘플링
→ 원본 비디오 다운로드
→ 프레임 추출
→ processed/train.jsonl 생성
→ [generate_sft] 질문 + 프레임 + 정답 → ANSWER / COT / LONG_COT 동시 생성
→ 최종 SFT JSON export (1행 → 3행)
```

## 2. 폴더 구조

```text
video_r1_sft_annotator/
├── README.md
├── requirements.txt
├── configs/
│   ├── generate_sft.yaml          ← 생성 단계 설정
│   ├── export_sft_generated.yaml  ← export 단계 설정
│   ├── generate_granularity.yaml  ← perspective 생성 단계 설정
│   ├── export_sft_perspective_generated.yaml ← perspective export 설정
│   ├── answer_style.yaml          ← (구버전 분류 파이프라인)
│   ├── reasoning_type.yaml        ← (구버전 분류 파이프라인)
│   └── export_sft.yaml            ← (구버전 분류 파이프라인)
├── scripts/
│   ├── prepare_video_r1_cot_data.sh
│   ├── generate_sft_data.sh       ← 생성 실행
│   ├── export_sft_generated.sh    ← export 실행
│   ├── generate_granularity_data.sh
│   ├── export_sft_perspective_generated.sh
│   ├── annotate_answer_style.sh   ← (구버전)
│   ├── annotate_reasoning_type.sh ← (구버전)
│   └── export_sft_dataset.sh      ← (구버전)
└── src/video_r1_sft_annotator/
    ├── __init__.py
    ├── annotate.py
    ├── export_sft_dataset.py
    ├── prepare_video_r1_cot.py
    ├── prompts.py
    └── utils.py
```

## 3. 각 파일의 역할

### `prepare_video_r1_cot.py`

`Video-R1-COT-165k.json`을 읽어서 annotation 대상 비디오 샘플을 준비한다.

하는 일:
- manifest 다운로드
- video row만 필터링
- subset 기준 샘플링
- 원본 비디오 다운로드
- 프레임 추출
- `processed/train.jsonl` 생성

### `annotate.py`

준비된 `processed/train.jsonl`을 읽고 모델 응답을 생성한다.

`annotation_task: generate_sft`로 설정하면 질문 + 프레임 + 정답을 주고, 모델이 아래 세 가지를 동시에 반환한다.

```json
{
  "answer": "직접 정답",
  "cot": "간결한 추론",
  "long_cot": "심층 추론"
}
```

출력 JSONL의 각 row:
- `question / options / gold_answer`
- `answer_raw / cot_raw / long_cot_raw`
- `frames` (프레임 경로 목록)
- `model_response` (모델 원문 응답)

### `export_sft_dataset.py`

`generated.jsonl`을 읽어서 최종 SFT JSON을 만든다.

입력 1행 → 출력 3행:

| `reasoning_depth` | `output` 형식 |
|---|---|
| `ANSWER` | `<ANSWER>\n...\n</ANSWER>` |
| `COT` | `<COT>\n...\n</COT>\n<ANSWER>\n...\n</ANSWER>` |
| `LONG_COT` | `<LONG_COT>\n...\n</LONG_COT>\n<ANSWER>\n...\n</ANSWER>` |

### `generate_granularity.yaml` + `export_sft_perspective_generated.yaml`

질문마다 하나의 추론 관점만 선택하는 perspective SFT 데이터 경로다.

- 생성 결과: `granularity_type` + `granularity_thinking_raw`
- export 결과:

```xml
<REASONING_TYPE>...</REASONING_TYPE>
<REASONING>...</REASONING>
<ANSWER>...</ANSWER>
```

### `prompts.py`

프롬프트를 정의한다.

- `build_generation_prompt()`: 세 가지 추론 동시 생성용 프롬프트
- `generation_system_prompt()`: 생성 태스크용 시스템 프롬프트
- `build_annotation_prompt()`: 구버전 분류용 프롬프트 (호환성 유지)

### `utils.py`

공통 유틸이다.

역할:
- JSON / JSONL 입출력
- Hugging Face dataset 다운로드
- archive 압축 해제
- 비디오 탐색 및 프레임 추출
- 질문 / 선택지 / 정답 파싱
- XML 태그 처리
- 모델 응답 JSON 파싱

## 4. 환경 설치

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

주의:
- `Qwen2.5-VL-72B-Instruct`를 사용하므로 큰 GPU 메모리가 필요하다.
- `vllm` 기반이라 Linux + CUDA 환경을 전제로 한다.

## 5. 데이터 준비

### 전체 준비

```bash
bash scripts/prepare_video_r1_cot_data.sh \
  --download-mode subset-directories
```

### 작은 샘플만 준비

```bash
bash scripts/prepare_video_r1_cot_data.sh \
  --sample-ratio 0.01 \
  --seed 42 \
  --download-mode subset-directories \
  --num-frames 8
```

### 일부 subset만 선택

```bash
bash scripts/prepare_video_r1_cot_data.sh \
  --subsets "LLaVA-Video-178K,STAR" \
  --sample-ratio 0.01 \
  --download-mode subset-directories
```

## 6. 데이터 준비 후 생성되는 파일

```text
data/video_r1_cot/raw/Video-R1-COT-165k.json
data/video_r1_cot/processed/train.jsonl
data/video_r1_cot/processed/frames/train/...
```

## 7. SFT 데이터 생성

### 1단계: ANSWER / COT / LONG_COT 동시 생성

```bash
bash scripts/generate_sft_data.sh
```

출력:
```text
outputs/generate_sft/generated.jsonl
outputs/generate_sft/summary.json
```

### 2단계: 최종 SFT JSON export

```bash
bash scripts/export_sft_generated.sh
```

출력:
```text
outputs/sft/video_r1_cot_sft.json
outputs/sft/summary.json
```

### Perspective SFT 데이터 생성

```bash
bash scripts/generate_granularity_data.sh
bash scripts/export_sft_perspective_generated.sh
```

출력:
```text
outputs/sft/video_r1_cot_perspective_sft.json
outputs/sft/perspective_summary.json
```

## 8. 설정 파일

### `configs/generate_sft.yaml`

주요 항목:
- `model_name_or_path`: 사용할 VLM
- `input_path`: `processed/train.jsonl` 경로
- `output_dir`: 생성 결과 저장 경로
- `annotation_task`: `generate_sft` 고정
- `frames_per_sample`: 모델에 넣을 프레임 수
- `temperature`: 생성 다양성 (기본 0.3)
- `max_completion_tokens`: 충분히 크게 설정 (기본 1024)

### `configs/export_sft_generated.yaml`

주요 항목:
- `generated_path`: `generated.jsonl` 경로
- `output_path`: 최종 JSON 저장 경로
- `summary_path`: 통계 저장 경로
- `max_samples`: 샘플 수 제한 (null이면 전체)

## 9. 최종 출력 형식

질문 하나당 3개의 훈련 샘플이 생성된다.

```json
{
  "instruction": "What is the person doing in the video?\n\nOptions:\n(A) Running\n(B) Walking\n(C) Sitting\n(D) Jumping",
  "input": "",
  "output": "<LONG_COT>\nIn the first few frames, the person is upright with both feet alternately leaving the ground...\nThis rapid leg movement combined with forward momentum indicates running.\nTherefore, the answer is A.\n</LONG_COT>\n<ANSWER>\nA\n</ANSWER>",
  "reasoning_depth": "LONG_COT",
  "question_id": "xxx",
  "video_id": "yyy",
  "source_subset": "LLaVA-Video-178K",
  "gold_answer": "A"
}
```

`reasoning_depth` 값: `ANSWER`, `COT`, `LONG_COT`

## 10. 가장 짧은 실행 순서

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1. 데이터 준비
bash scripts/prepare_video_r1_cot_data.sh \
  --sample-ratio 0.01 \
  --download-mode subset-directories \
  --num-frames 8

# 2. 세 가지 추론 생성
bash scripts/generate_sft_data.sh

# 3. SFT JSON export
bash scripts/export_sft_generated.sh
```

최종 결과:

```text
outputs/sft/video_r1_cot_sft.json
```

## 11. 한 줄 요약

```text
Video-R1-COT-165k를 받아서,
비디오 프레임 기반으로 ANSWER / COT / LONG_COT 세 가지 추론을 생성하고,
최종적으로 SFT 가능한 JSON 데이터셋까지 만드는 전용 파이프라인
```
