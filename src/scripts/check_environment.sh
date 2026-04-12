#!/bin/bash
#
# GRPO 학습 전 환경 점검 — 이 스크립트가 통과하면(원하는 모드에서) 세팅 OK로 보면 된다.
#
# 사용:
#   cd /path/to/GRPO_Video_2
#   source scripts/hpc_activate_grpo.sh      # GRPO 전용 venv ($HOME/scratch/.venv_grpo)
#   # 또는: source scripts/hpc_activate_realign.sh
#   bash src/scripts/check_environment.sh
#
# 로그인 노드 등 GPU 없음: GPU/torch.cuda/flash_attn 실패를 막지 않게 하려면
#   GRPO_CHECK_NO_GPU=1 bash src/scripts/check_environment.sh
#
# 기본 merged 모델 경로는 run_grpo_answer_only_lora.sh 의 QWEN_PATH 와 맞춤.
# 다른 경로면:  MERGED_MODEL_DIR=sft/outputs/... bash src/scripts/check_environment.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

GRPO_CHECK_NO_GPU_LC="$(printf '%s' "${GRPO_CHECK_NO_GPU:-0}" | tr '[:upper:]' '[:lower:]')"
GPU_STRICT=true
if [[ "${GRPO_CHECK_NO_GPU_LC}" == "1" || "${GRPO_CHECK_NO_GPU_LC}" == "true" || "${GRPO_CHECK_NO_GPU_LC}" == "yes" ]]; then
  GPU_STRICT=false
fi

MERGED_MODEL_DIR="${MERGED_MODEL_DIR:-sft/outputs/qwen25vl7b_lora_merged_length}"

echo "================================================"
echo "GRPO_Video_2 Environment Check"
echo "REPO_ROOT=${REPO_ROOT}"
echo "GPU_STRICT=${GPU_STRICT}  (set GRPO_CHECK_NO_GPU=1 on login nodes without GPU)"
echo "================================================"
echo ""

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ALL_CHECKS_PASSED=true

print_check() {
    if [ "$1" -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $2"
    else
        echo -e "${RED}✗${NC} $2"
        ALL_CHECKS_PASSED=false
    fi
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

echo "1. Checking Python Environment..."
echo "-----------------------------------"

if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
    PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

    if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 11 ]; then
        print_check 0 "Python $PYTHON_VERSION (>= 3.11)"
    else
        print_check 1 "Python version: $PYTHON_VERSION (need >= 3.11)"
    fi
else
    print_check 1 "Python not found"
fi

if command -v conda &> /dev/null; then
    print_check 0 "Conda installed"
else
    print_warning "Conda not found (optional)"
fi

echo ""
echo "2. Checking GPU Setup..."
echo "-----------------------------------"

if command -v nvidia-smi &> /dev/null; then
    print_check 0 "NVIDIA drivers installed (nvidia-smi)"
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "   Found $GPU_COUNT GPU(s):"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | while read -r line; do
        echo "   - GPU $line"
    done
    MIN_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | head -1)
    if [ "$MIN_VRAM" -ge 20000 ]; then
        print_check 0 "Minimum VRAM: ${MIN_VRAM}MB (>= 20GB)"
    else
        print_check 1 "Minimum VRAM: ${MIN_VRAM}MB (need >= 20GB, recommend 24GB+)"
    fi
else
    if ${GPU_STRICT}; then
        print_check 1 "NVIDIA drivers not found (nvidia-smi not available)"
    else
        print_warning "nvidia-smi not available (skipped: GRPO_CHECK_NO_GPU=1)"
    fi
fi

if command -v nvcc &> /dev/null; then
    NVCC_VERSION=$(nvcc --version | awk -F'release ' '/release/{print $2}' | awk -F',' '{print $1}')
    if [[ "${NVCC_VERSION}" == "12.4" ]]; then
        print_check 0 "nvcc release ${NVCC_VERSION} (expected 12.4)"
    else
        print_warning "nvcc release ${NVCC_VERSION} (runbook target: 12.4)"
    fi
else
    print_warning "nvcc not found (CUDA toolkit check skipped)"
fi

if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null)
    print_check 0 "PyTorch CUDA available (version: $CUDA_VERSION)"
else
    if ${GPU_STRICT}; then
        print_check 1 "PyTorch CUDA not available"
    else
        print_warning "PyTorch CUDA not available (skipped: GRPO_CHECK_NO_GPU=1)"
    fi
fi

echo ""
echo "3. Checking Python Packages..."
echo "-----------------------------------"

check_package() {
    if python -c "import $1" 2>/dev/null; then
        VERSION=$(python -c "import $1; print($1.__version__ if hasattr($1, '__version__') else 'unknown')" 2>/dev/null)
        print_check 0 "$1 ($VERSION)"
        return 0
    else
        print_check 1 "$1 not installed"
        return 1
    fi
}

check_package_optional_flash() {
    if python -c "import flash_attn" 2>/dev/null; then
        VERSION=$(python -c "import flash_attn; print(flash_attn.__version__)" 2>/dev/null)
        print_check 0 "flash_attn ($VERSION)"
    else
        if ${GPU_STRICT}; then
            print_check 1 "flash_attn not installed"
        else
            print_warning "flash_attn not importable (common on login without GPU build; OK on GPU nodes if installed)"
        fi
    fi
}

check_package "torch"
check_package "transformers"
check_package "peft"
check_package "trl"
check_package "datasets"
# yt-dlp: 일부 데이터 준비 스크립트용; 프레임/JSONL이 이미 있으면 GRPO 학습에는 불필요
if python -c "import yt_dlp" 2>/dev/null; then
    print_check 0 "yt_dlp (optional, data prep)"
else
    print_warning "yt_dlp not installed (optional if frames/JSONL already prepared)"
fi
check_package "vllm"
check_package "deepspeed"
check_package "accelerate"
check_package_optional_flash

echo ""
echo "pip check (r1-v는 setup.sh에서 --no-deps 설치 → liger-kernel 줄은 video GRPO에 필수 아님):"
PIP_CHECK_OUT=""
PIP_CHECK_OUT=$(pip check 2>&1 || true)
echo "   ${PIP_CHECK_OUT}"
if echo "$PIP_CHECK_OUT" | grep -q '^No broken requirements found\.$'; then
    print_check 0 "pip check: OK"
else
    NON_LIGER=$(echo "$PIP_CHECK_OUT" | grep -v -i liger | grep -v '^[[:space:]]*$' || true)
    if [[ -n "$NON_LIGER" ]]; then
        print_check 1 "pip check: ${PIP_CHECK_OUT}"
    else
        print_warning "pip check: liger-kernel만 언급 — r1-v 메타데이터; open_r1 video GRPO는 무시 가능"
        print_check 0 "pip check: OK for GRPO"
    fi
fi

echo ""
echo "Version compatibility checks (runbook target):"
python - <<'PY' 2>/dev/null || true
import importlib

targets = {
    "torch": "2.5.1+cu124",
    "transformers": None,
    "peft": "0.14.0",
    "trl": "0.14.0",
    "deepspeed": "0.15.4",
    "vllm": "0.7.2",
    "flash_attn": "2.6.3",
}

for pkg, target in targets.items():
    try:
        mod = importlib.import_module(pkg)
        got = getattr(mod, "__version__", "unknown")
        if target is None:
            print(f"  - {pkg}: {got}")
        elif got == target:
            print(f"  - {pkg}: {got} (OK)")
        else:
            print(f"  - {pkg}: {got} (target {target})")
    except Exception:
        print(f"  - {pkg}: not installed")
PY

echo ""
echo "Optional packages:"
if python -c "import wandb" 2>/dev/null; then
    print_check 0 "wandb"
else
    print_warning "wandb not installed (optional)"
fi

echo ""
echo "4. Checking Disk Space..."
echo "-----------------------------------"

DISK_AVAILABLE_HUMAN=$(df -h . | awk 'NR==2 {print $4}')
DISK_AVAILABLE_KB=$(df -Pk . | awk 'NR==2 {print $4}')

if [[ -n "${DISK_AVAILABLE_KB}" && "${DISK_AVAILABLE_KB}" =~ ^[0-9]+$ ]]; then
    DISK_AVAILABLE_GB=$((DISK_AVAILABLE_KB / 1024 / 1024))
else
    DISK_AVAILABLE_GB=""
fi

echo "   Available disk space: ${DISK_AVAILABLE_HUMAN}"
if [[ -n "${DISK_AVAILABLE_GB}" ]]; then
    if [ "${DISK_AVAILABLE_GB}" -ge 100 ]; then
        print_check 0 "Disk space: ${DISK_AVAILABLE_GB}GB (>= 100GB)"
    else
        print_warning "Disk space: ${DISK_AVAILABLE_GB}GB (recommend >= 100GB)"
    fi
else
    print_warning "Unable to parse disk space in GB"
fi

echo ""
echo "5. Checking Repository Structure..."
echo "-----------------------------------"

check_path() {
    if [ -e "$1" ]; then
        print_check 0 "$1"
        return 0
    else
        print_check 1 "$1 not found"
        return 1
    fi
}

check_path "src/r1-v/src/open_r1/grpo.py"
check_path "src/scripts/run_grpo_answer_only_lora.sh"
check_path "sft/data/prepare_/prepare_all_grpo_data.sh"
check_path "sft/data/prepare_/prepare_uvb_full_split_local_videos.sh"
check_path "sft/data/prepare_/prepare_uvb_grpo_data.sh"
check_path "sft/data/prepare_/prepare_videommmu_grpo_data.sh"
check_path "sft/data/prepare_/prepare_mmvu_grpo_data.sh"
check_path "sft/data/prepare_/prepare_video_r1_grpo_data.sh"
check_path "src/r1-v/configs/zero1_no_optimizer.json"

echo ""
echo "Merged model (QWEN_PATH default): ${MERGED_MODEL_DIR}"
if [ -d "${MERGED_MODEL_DIR}" ]; then
    check_path "${MERGED_MODEL_DIR}/config.json"
    check_path "${MERGED_MODEL_DIR}/tokenizer.json"
    check_path "${MERGED_MODEL_DIR}/tokenizer_config.json"
    check_path "${MERGED_MODEL_DIR}/preprocessor_config.json"
else
    print_warning "Merged model dir not found: ${MERGED_MODEL_DIR} (SFT merge 후 생성 또는 MERGED_MODEL_DIR 지정)"
fi

echo ""
echo "6. Checking Network Access..."
echo "-----------------------------------"

if curl -s --head --connect-timeout 5 https://huggingface.co 2>/dev/null | head -n 1 | grep -q "HTTP"; then
    print_check 0 "HuggingFace reachable"
else
    print_warning "Cannot reach huggingface.co (offline 캐시만 쓸 경우 무시 가능)"
fi

echo ""
echo "================================================"
if [ "$ALL_CHECKS_PASSED" = true ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    echo "GRPO 학습 전 이 스크립트만 통과하면 된다고 보면 됩니다 (GPU 노드에서는 GRPO_CHECK_NO_GPU 미설정 권장)."
    echo ""
    echo "Next:"
    echo "  bash src/scripts/run_grpo_answer_only_lora.sh"
else
    echo -e "${RED}✗ Some checks failed${NC}"
    echo ""
    echo "Common fixes:"
    echo "  - GRPO venv: bash scripts/run_setup_grpo.sh"
    echo "  - 로그인 노드만 확인: GRPO_CHECK_NO_GPU=1 bash src/scripts/check_environment.sh"
    echo "  - GPU: nvidia-smi, module load cuda"
fi
echo "================================================"

exit $([ "$ALL_CHECKS_PASSED" = true ] && echo 0 || echo 1)
