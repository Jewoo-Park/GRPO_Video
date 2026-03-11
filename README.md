# GRPO_Video 레포지토리 구조 및 상세 리뷰

이 문서는 레포 전체를 한 번에 파악할 수 있도록 **디렉터리 구조**, **데이터 흐름**, **학습/평가 파이프라인**, **핵심 코드**를 정리한 리뷰입니다.

---

## 1. 전체 디렉터리 개요

```
GRPO_Video/
├── setup.sh                    # 레포 공통 환경 설치 (r1-v, vllm, deepspeed, flash-attn, trl, peft 등)
├── merge_readme.md             # SFT LoRA merge 가이드 (한국어)
├── analyze_train_log.ipynb    # GRPO 학습 로그 분석용 노트북
├── parsed_train_metrics.csv   # 파싱된 학습 메트릭 (분석 결과)
│
├── data/
│   └── urban_video_bench/      # Urban Video Bench (UVB) 데이터
│       ├── grpo/               # GRPO 학습/테스트용 JSONL (uvb_grpo_train.jsonl, uvb_grpo_test.jsonl)
│       └── processed/         # 파이프라인 산출물: train_80.jsonl, test_20.jsonl, frames/{train,test}/
│
├── sft/                        # SFT 전용 파이프라인 (Qwen2.5-VL-3B LoRA)
│   ├── configs/               # train_lora_qwen25vl3b.yaml, merge_lora_qwen25vl3b.yaml
│   ├── data/                  # SFT용 JSON (aqua_rat_multiple_choice*.json, aqua_rat_open_form*.json)
│   ├── scripts/               # train_sft.py, merge_lora.py, run_train.sh, run_merge.sh
│   ├── outputs/               # LoRA 어댑터 및 merge 결과 (qwen25vl3b_lora_sft_40, qwen25vl3b_lora_merged_*)
│   └── README.md
│
├── src/
│   ├── eval/                  # UVB 데이터 준비 및 오프라인 평가
│   │   ├── prepare_urban_video_bench.py   # HF 데이터셋 → JSONL 내보내기
│   │   ├── prepare_uvb_pipeline.py         # 샘플링·분할·비디오 다운로드·프레임 추출
│   │   ├── uvb_to_grpo.py                 # processed → GRPO 형식 JSONL 변환
│   │   └── uvb_eval_only.py               # vLLM으로 테스트셋 추론 + 정확도/포맷 통계
│   │
│   ├── r1-v/                  # GRPO 학습 코드 (open_r1 기반)
│   │   ├── configs/           # DeepSpeed 설정 (zero1_no_optimizer.json)
│   │   ├── src/open_r1/
│   │   │   ├── grpo_uvb.py    # UVB 전용 GRPO 진입점 (데이터 로드, 리워드, 트레이너 호출)
│   │   │   └── trainer/      # Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainerModified
│   │   └── outputs/          # GRPO 체크포인트 및 test_predictions.jsonl
│   │
│   └── scripts/               # 실행·환경 체크·문서
│       ├── run_grpo_uvb_answer_only.sh      # GRPO UVB 실행 (기본)
│       ├── run_grpo_uvb_answer_only_lora.sh
│       ├── run_sft_grpo_a100x2.sh           # SFT → Merge → GRPO 한 번에
│       ├── prepare_uvb_dataset.sh
│       ├── prepare_uvb_40_split_download_frames.sh
│       ├── prepare_uvb_grpo_data.sh
│       ├── check_environment.sh
│       └── RUN_GRPO_UVB.md
│
├── docs/                      # 분석/문서 (예: UVB_GRPO_performance_analysis.md)
└── analysis_plots/            # 학습 분석 결과 (training_summary.md 등)
```

---

## 2. 파이프라인 요약 (SFT → Merge → GRPO → Eval)

| 단계 | 목적 | 스크립트/모듈 | 입력 | 출력 |
|------|------|----------------|------|------|
| **데이터 준비** | UVB 메타데이터·프레임·GRPO JSONL 생성 | `prepare_urban_video_bench.py` → `prepare_uvb_pipeline.py` → `uvb_to_grpo.py` | HF `EmbodiedCity/UrbanVideo-Bench` | `data/urban_video_bench/grpo/uvb_grpo_{train,test}.jsonl` |
| **SFT** | Qwen2.5-VL-3B에 LoRA SFT | `sft/scripts/run_train.sh` → `train_sft.py` | `sft/data/*.json`, `train_lora_qwen25vl3b.yaml` | `sft/outputs/qwen25vl3b_lora_sft_40/` (어댑터) |
| **Merge** | LoRA를 백본에 병합 | `sft/scripts/run_merge.sh` → `merge_lora.py` | `merge_lora_qwen25vl3b.yaml` (base + adapter 경로) | `sft/outputs/qwen25vl3b_lora_merged_from_sft40/` (전체 모델) |
| **GRPO** | UVB 비디오 QA에 GRPO 학습 | `run_grpo_uvb_answer_only.sh` → `open_r1.grpo_uvb` | `uvb_grpo_train.jsonl`, `uvb_grpo_test.jsonl`, merged 모델 | `src/r1-v/outputs/uvb_grpo_answer_only/` (체크포인트 + `test_predictions.jsonl`) |
| **Eval (오프라인)** | 저장된 모델로 테스트셋 정확도·포맷 측정 | `uvb_eval_only.py` | `--model`, `--test-file` | 터미널 메트릭 + 선택 시 `--save-preds` / `--save-json` |

**한 번에 실행 (SFT → Merge → GRPO):**

```bash
bash src/scripts/run_sft_grpo_a100x2.sh
```

GRPO만 다시 돌리려면:

```bash
DO_SFT=false DO_MERGE=false bash src/scripts/run_sft_grpo_a100x2.sh
```

이때 merged 모델은 `GRPO_QWEN_PATH`로 지정하거나, 스크립트 내 `resolve_sft_merged_model()`이 찾은 경로를 사용합니다.

---

## 3. 데이터 형식 상세

### 3.1 GRPO용 JSONL (`data/urban_video_bench/grpo/`)

- **파일:** `uvb_grpo_train.jsonl`, `uvb_grpo_test.jsonl`
- **생성:** `uvb_to_grpo.py` (입력: `processed/train_80.jsonl`, `test_20.jsonl`, `processed/frames/`)

각 줄은 한 샘플(한 비디오·한 질문)을 나타내는 JSON 객체:

| 필드 | 설명 |
|------|------|
| `video_id` | 비디오 식별자 (파일명 등) |
| `question_id` | 질문 ID |
| `question_category` | 질문 카테고리 |
| `problem` | 프롬프트에 넣을 질문 문자열 (예: `"Question: ..."`) |
| `frames` | **상대 경로** 리스트. GRPO/트레이너는 이 경로를 JSONL 파일이 있는 디렉터리(`grpo/`) 기준으로 해석 |
| `solution` | 정답. 현재 `uvb_to_grpo.py`는 `f"<answer>{answer}</answer>"` 형태로 저장 (소문자 태그). 학습/평가 쪽은 `<ANSWER>...</ANSWER>`(대문자) 패턴도 처리 가능하도록 되어 있으나, **형식 통일을 위해 `<ANSWER>...</ANSWER>`로 저장하는 것을 권장** |

### 3.2 Processed UVB (`data/urban_video_bench/processed/`)

- `train_80.jsonl`, `test_20.jsonl`: 비디오/질문 메타데이터 (예: `video_id`, `question`, `answer`, `question_category`)
- `frames/train/<video_stem>/frame_*.jpg`, `frames/test/...`: 추출된 프레임 이미지
- `uvb_to_grpo.py`는 `safe_stem(video_id)`로 프레임 디렉터리 이름을 맞추고, `frames` 필드를 GRPO 출력 디렉터리 기준 **상대 경로**로 씁니다.

### 3.3 SFT 데이터 (`sft/data/`)

- **형식:** `instruction` / `input` / `output` 리스트가 있는 JSON (ARM 레포 SFT 형식)
- **파일:** `aqua_rat_multiple_choice.json`, `aqua_rat_open_form.json` (및 `*_40.json` 등)
- **출력 태그:** SFT 설정에서 `answer`, `cot`, `long_cot` 지원 → 각각 `<ANSWER>...</ANSWER>`, `<COT>...</COT> + `<ANSWER>...</ANSWER>`, `<LONG_COT>...</LONG_COT> + `<ANSWER>...</ANSWER>` 형식으로 학습

---

## 4. 핵심 코드 설명

### 4.1 `grpo_uvb.py` (GRPO UVB 진입점)

- **역할:** UVB용 GRPO 학습 전용. TRL `TrlParser`로 `GRPOUVBScriptArguments`, `GRPOConfig`, `ModelConfig` 파싱.
- **데이터 로드:** `datasets.load_dataset("json", data_files={"train": train_file, "test": test_file})`. `frames` 경로는 JSONL 디렉터리 기준으로 절대 경로로 해석되도록 `resolve_frames_for_split`에서 변환.
- **프롬프트:** `SYSTEM_PROMPT`로 “simple → answer only, complex → COT/LONG_COT 후 ANSWER” 규칙과 예시를 고정. 사용자 메시지는 `[이미지 토큰들] + problem` 형태의 대화로 구성.
- **리워드:**
  - `answer_accuracy`: 생성 답과 `solution`의 **선지 글자(A–G)** 비교 (`_extract_choice_letter`).
  - `answer_format`: `<ANSWER>...</ANSWER>` 단독, 또는 `<COT>...</COT> + `<ANSWER>...`, `<LONG_COT>...</LONG_COT> + `<ANSWER>...` 중 하나에 맞으면 1, 아니면 0.
- **트레이너:** `use_vllm` 여부에 따라 `Qwen2VLGRPOTrainer` 또는 `Qwen2VLGRPOVLLMTrainerModified` 선택. 학습 종료 후 테스트셋이 있으면 `run_test_inference()`로 추론하고 `test_predictions.jsonl` 저장 (필드: `video_id`, `question_id`, `pred_raw`, `pred_answer`, `gt_answer`, `correct`, `format_ok` 등).

### 4.2 `uvb_eval_only.py` (오프라인 평가)

- **역할:** 이미 저장된 모델 + vLLM으로 테스트 JSONL만 읽어 추론 후 정확도·포맷·리즈닝 통계 출력.
- **입력:** `--model`, `--test-file`. 프레임 경로는 테스트 JSONL 파일의 부모 디렉터리 기준으로 해석.
- **시스템 프롬프트:** `grpo_uvb.py`의 `SYSTEM_PROMPT`와 동일한 규칙/예시 문자열을 하드코딩.
- **메트릭:** `answer_accuracy`, `answer_format_rate`, `reasoning_present_rate`, 평균 completion/reasoning 길이(문자/단어/토큰), `reasoning_type_counts` (none / cot_tag / long_cot_tag).
- **출력:** 터미널에 메트릭 출력; `--save-preds`로 예측 JSONL, `--save-json`으로 메트릭+결과 통합 JSON 저장.

### 4.3 `uvb_to_grpo.py` (processed → GRPO JSONL)

- **역할:** `processed/train_80.jsonl`, `test_20.jsonl`과 `processed/frames/`를 사용해 GRPO용 `uvb_grpo_train.jsonl`, `uvb_grpo_test.jsonl` 생성.
- **프레임 경로:** `frames_root / split_name / safe_stem(video_id)` 아래 `frame_*.jpg`를 정렬한 뒤, **출력 디렉터리(`output_dir`) 기준 상대 경로**로 저장. 따라서 GRPO 실행 시 작업 디렉터리/경로 해석이 `grpo/`를 기준으로 하도록 맞춰져 있음.
- **solution:** `item["answer"]`를 `f"<answer>{answer}</answer>"`로 감싸서 저장. 위에서 언급한 대로, 학습/평가와의 형식 통일을 위해 `<ANSWER>...</ANSWER>`로 바꾸는 것이 좋음.

### 4.4 SFT 설정 (`sft/configs/`)

- **train_lora_qwen25vl3b.yaml:**  
  - `train_files`: `sft/data/` 내 JSON 목록  
  - LoRA target: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`  
  - `reasoning_formats`: `[answer, cot, long_cot]`, `format_mix_strategy: expand`, `drop_code_cot: true`
- **merge_lora_qwen25vl3b.yaml:**  
  - `model_name_or_path`, `adapter_name_or_path`, `export_dir`  
  - **`remap_adapter_keys: true` 필수** (키 불일치 시 LoRA가 적용되지 않음).

### 4.5 GRPO 실행 스크립트 (`run_grpo_uvb_answer_only.sh`)

- **작업 디렉터리:** `src/r1-v`로 이동 후 `python -m open_r1.grpo_uvb` 실행. 따라서 `TRAIN_FILE`/`TEST_FILE` 기본값 `../../data/urban_video_bench/grpo/...`는 **레포 루트 기준이 아니라 `src/r1-v` 기준** 상대 경로.
- **주요 환경 변수:** `QWEN_PATH`, `TRAIN_FILE`, `TEST_FILE`, `OUTPUT_DIR`, `NUM_GPUS`, `TRAIN_NUM_GPUS`(vLLM 사용 시 학습용 GPU 수), `MAX_PIXELS`, `NUM_GENERATIONS`, `USE_PEFT`, `LORA_*`, `VLLM_GPU_UTIL`, `DS_CONFIG` 등.
- vLLM 사용 시: 트레이너는 학습에 `TRAIN_NUM_GPUS`만 쓰고, 별도 1 GPU를 vLLM용으로 둡니다 (`NUM_GPUS > 1`일 때).

---

## 5. 답변/포맷 파싱 규칙 (통일 사항)

- **허용 태그:** `<ANSWER>...</ANSWER>` (및 오타 대비 `<ANSWERS>...</ANSWERS>`). 파싱은 **대문자 태그** 기준으로 맞춰져 있음.
- **리즈닝:** `<COT>...</COT>`, `<LONG_COT>...</LONG_COT>`.
- **선지 추출:** UVB는 주로 A–E 다지선. `_extract_choice_letter`는 답 텍스트에서 A–G 글자 하나를 찾아 소문자로 정규화해 비교.
- **권장:** 데이터 생성·라벨(`solution`)은 모두 **대문자 태그** (`<ANSWER>`, `<COT>`, `<LONG_COT>`)만 사용하면 SFT/GRPO/평가 간 동작이 일관됩니다.

---

## 6. 환경 및 트러블슈팅 요약

- **설치:** `setup.sh` (레포 루트에서 실행). `src/r1-v` editable 설치, vllm 0.7.2, deepspeed 0.15.4, trl 0.14.0, flash-attn 등.  
  - torch 업그레이드 후 flash-attn 심볼 오류가 나면: merge 단계에서는 flash-attn 제거 후 진행 가능; GRPO 전에 현재 torch에 맞춰 flash-attn 재설치.
- **환경 검증:** `src/scripts/check_environment.sh`로 Python, GPU, torch, vllm, deepspeed, 주요 스크립트 경로 확인.
- **Merge:** `remap_adapter_keys: true`, `export_dir`는 디스크 여유 있는 경로(다른 볼륨 등) 권장. 자세한 내용은 `merge_readme.md` 참고.
- **GRPO 디바이스 오류:** 이전에 “Attention bias and Q/K/V should be on the same device” 같은 오류는 트레이너/vLLM 쪽에서 디바이스 일치하도록 수정된 상태입니다.

---

## 7. 문서·추가 자료

| 문서 | 내용 |
|------|------|
| `merge_readme.md` | SFT LoRA merge 단계별 가이드, flash-attn 이슈, 트러블슈팅 표 |
| `src/scripts/RUN_GRPO_UVB.md` | GRPO 실행 방법, 환경 변수, 데이터 파이프라인 참고 |
| `sft/README.md` | SFT 전용 파이프라인, 데이터 형식, reasoning_formats 설명 |
| `docs/UVB_GRPO_performance_analysis.md` | UVB GRPO 성능 분석 (있는 경우) |

---

이 문서는 레포 루트의 **REPO_STRUCTURE_AND_REVIEW.md**로 저장되어 있으며, 디렉터리 구조·데이터·파이프라인·핵심 코드·형식 규칙·환경을 한 곳에서 참고할 수 있도록 구성되어 있습니다.
