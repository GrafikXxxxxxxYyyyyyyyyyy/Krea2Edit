#!/usr/bin/env bash
# Развёртывание на свежем GPU-инстансе (vast.ai, runpod, любой Ubuntu с CUDA).
#
#   export HF_TOKEN=hf_...
#   bash bootstrap.sh
#
# Скрипт идемпотентен: повторный запуск ничего не сломает.

set -euo pipefail

echo "=== 1/5 проверка окружения ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || {
    echo "ОШИБКА: nvidia-smi не найден — это точно GPU-инстанс?"; exit 1; }

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ОШИБКА: HF_TOKEN не задан."
    echo "  Репозитории krea/Krea-2-* gated: прими условия на странице модели,"
    echo "  создай токен с правом read и выполни: export HF_TOKEN=hf_..."
    exit 1
fi

echo
echo "=== 2/5 кэш HuggingFace ==="
# Кладём кэш на большой диск инстанса, а не в / — веса весят ~30 ГБ.
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
mkdir -p "$HF_HOME"
echo "HF_HOME=$HF_HOME"
df -h "$HF_HOME" | tail -1

echo
echo "=== 3/5 зависимости ==="
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
python -c "import torch, diffusers, transformers, gradio
print(f'  torch {torch.__version__}, cuda={torch.cuda.is_available()}')
print(f'  diffusers {diffusers.__version__}')
print(f'  transformers {transformers.__version__}')
print(f'  gradio {gradio.__version__}')
from diffusers import Krea2Pipeline
print('  Krea2Pipeline доступен')"

echo
echo "=== 4/5 прогрев весов (долго: ~30 ГБ) ==="
# Качаем ДО запуска UI: иначе первый запрос в Gradio висит без объяснений.
python - <<'PY'
import os, time, torch
from krea2_studio import Krea2EditPipeline
from transformers import AutoProcessor
mid = os.environ.get("KREA2_MODEL", "krea/Krea-2-Turbo")
t0 = time.time()
Krea2EditPipeline.from_pretrained(mid, dtype=torch.bfloat16)
AutoProcessor.from_pretrained(os.environ.get("KREA2_PROCESSOR", "Qwen/Qwen3-VL-4B-Instruct"))
print(f"  веса в кэше за {(time.time()-t0)/60:.1f} мин")
PY

echo
echo "=== 5/5 диагностика ==="
python smoke_test.py --steps 4

echo
echo "Готово. Запуск интерфейса:"
echo "  python app.py                  # порт 7860, нужен проброс порта"
echo "  GRADIO_SHARE=1 python app.py   # публичная ссылка *.gradio.live"
