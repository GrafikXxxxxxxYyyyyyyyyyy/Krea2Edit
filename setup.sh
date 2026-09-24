#!/usr/bin/env bash
# Установка и запуск krea2-studio в рабочей конфигурации:
#
#   трансформер   DAF-K2T (CivitAI, few-step файнтюн Krea 2)
#   энкодер       DreamFast/Qwen3-VL-4b-Heretic
#   edit-LoRA     conradlocke/krea2-identity-edit
#   интерфейс     Gradio с публичной ссылкой *.gradio.live
#
#   bash setup.sh                 # установить что нужно и запустить
#   bash setup.sh --no-launch     # только установить
#   bash setup.sh --reset-tokens  # заново спросить токены и пароль
#
# Токены спрашиваются один раз и сохраняются в .env рядом со скриптом (права 600,
# в git не попадает). Повторный запуск ничего не качает заново — сразу запуск.
# Любой параметр из .env можно переопределить переменной окружения, например
#   KREA2_EDIT_LORA=r64 bash setup.sh

set -euo pipefail
cd "$(dirname "$0")"
ROOT=$PWD
ENV_FILE="$ROOT/.env"

LAUNCH=1
RESET=0
for arg in "$@"; do
    case "$arg" in
        --no-launch) LAUNCH=0 ;;
        --reset-tokens) RESET=1 ;;
        -h|--help) sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "неизвестный аргумент: $arg (см. --help)"; exit 2 ;;
    esac
done

say()  { echo -e "\n=== $* ==="; }
die()  { echo "ОШИБКА: $*" >&2; exit 1; }

# Значения из .env, если он уже есть. Переменные окружения важнее файла.
if [ -f "$ENV_FILE" ]; then
    while IFS='=' read -r key value; do
        [[ "$key" =~ ^[A-Z0-9_]+$ ]] || continue
        [ -n "${!key+x}" ] && continue                 # уже задано снаружи
        value="${value%\"}"; value="${value#\"}"
        export "$key=$value"
    done < "$ENV_FILE"
fi
if [ "$RESET" = 1 ]; then
    unset HF_TOKEN CIVITAI_TOKEN GRADIO_AUTH GRADIO_AUTH_ASKED
fi

# ---------- конфигурация по умолчанию ----------
DAF_VERSION_ID=3094215                                  # civitai.com/models/2750471, v1
export KREA2_MODELS_DIR="${KREA2_MODELS_DIR:-${WORKSPACE:-$ROOT}/models}"
export KREA2_MODEL="${KREA2_MODEL:-krea/Krea-2-Turbo}"
export KREA2_TRANSFORMER="${KREA2_TRANSFORMER-$KREA2_MODELS_DIR/dafK2T_v1.safetensors}"
export KREA2_DISTILLED="${KREA2_DISTILLED:-1}"         # DAF-K2T — few-step, без CFG
export KREA2_TEXT_ENCODER="${KREA2_TEXT_ENCODER-DreamFast/Qwen3-VL-4b-Heretic}"
export KREA2_PROCESSOR="${KREA2_PROCESSOR:-Qwen/Qwen3-VL-4B-Instruct}"
export KREA2_EDIT_LORA="${KREA2_EDIT_LORA:-conradlocke/krea2-identity-edit}"
export HF_HOME="${HF_HOME:-${WORKSPACE:-$HOME}/.hf_home}"
export PORT="${PORT:-7860}"

# ---------- 1. окружение ----------
say "1/6 окружение"
command -v nvidia-smi >/dev/null || die "nvidia-smi не найден — нужен GPU-инстанс"
GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
echo "GPU: $GPU"

# Трансформер в bf16 — 24 ГБ, энкодер — 8, плюс VAE и LoRA: всего ~36 ГБ.
if [ -z "${KREA2_OFFLOAD:-}" ]; then
    if [ "$VRAM_MB" -lt 40000 ]; then
        export KREA2_OFFLOAD=1
        echo "VRAM меньше 40 ГБ — включаю CPU-offload (медленнее, но влезет)"
    else
        export KREA2_OFFLOAD=0
    fi
fi

if [ -z "${VIRTUAL_ENV:-}" ] && [ -f /venv/main/bin/activate ]; then
    source /venv/main/bin/activate                      # venv vast.ai-образов
fi
command -v python >/dev/null || die "python не найден"
echo "python: $(command -v python) ($(python --version 2>&1))"

mkdir -p "$KREA2_MODELS_DIR" "$HF_HOME"
FREE_GB=$(df -Pk "$KREA2_MODELS_DIR" | awk 'NR==2 {print int($4/1024/1024)}')
echo "свободно на диске: ${FREE_GB} ГБ (нужно ~25 ГБ при первой установке)"

# ---------- 2. зависимости ----------
say "2/6 зависимости"
if python -c "from diffusers import Krea2Pipeline; import peft, gradio, transformers, safetensors" 2>/dev/null; then
    echo "уже установлены"
else
    if command -v uv >/dev/null && [ -n "${VIRTUAL_ENV:-}" ]; then
        uv pip install -r requirements.txt
    else
        python -m pip install -r requirements.txt
    fi
fi
python -c "import torch, diffusers, transformers, gradio
print(f'torch {torch.__version__} (cuda {torch.cuda.is_available()}), diffusers {diffusers.__version__}, '
      f'transformers {transformers.__version__}, gradio {gradio.__version__}')"

# ---------- 3. токены ----------
say "3/6 токены"
[ -t 0 ] && INTERACTIVE=1 || INTERACTIVE=0

check_hf() {
    HF_TOKEN="$1" python - "$KREA2_MODEL" <<'PY'
import sys
from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url
from huggingface_hub.errors import GatedRepoError
repo = sys.argv[1]
try:
    name = HfApi().whoami()["name"]
except Exception as e:
    print(f"  токен не принят: {type(e).__name__}"); sys.exit(1)
try:
    get_hf_file_metadata(hf_hub_url(repo, "model_index.json"))
except GatedRepoError:
    print(f"  вошли как {name}, но доступа к {repo} нет.\n"
          f"  Откройте https://huggingface.co/{repo} и примите условия, затем запустите снова.")
    sys.exit(2)
print(f"  HF: вошли как {name}, доступ к {repo} есть")
PY
}

check_civitai() {
    # Range CivitAI игнорирует, поэтому смотрим только статус и обрываем по таймауту.
    local status
    status=$(curl -s -L -o /dev/null -D - --max-time 15 \
                  -H "Authorization: Bearer $1" \
                  "https://civitai.com/api/download/models/$DAF_VERSION_ID" 2>/dev/null \
             | awk '/^HTTP/ {code=$2} END {print code}' || true)
    case "$status" in
        200) echo "  CivitAI: доступ к DAF-K2T есть"; return 0 ;;
        401|403) echo "  CivitAI ответил $status. Токен неверный, или в настройках аккаунта не включён"
                 echo "  показ Mature-контента (civitai.com/user/account -> Content Moderation):"
                 echo "  модель помечена NSFW."; return 1 ;;
        *) echo "  CivitAI ответил '${status:-нет ответа}'"; return 1 ;;
    esac
}

ask_token() {   # ask_token ПЕРЕМЕННАЯ "подсказка" функция_проверки
    local var=$1 hint=$2 check=$3 value rc
    if [ -n "${!var:-}" ]; then
        rc=0; $check "${!var}" || rc=$?
        [ "$rc" = 0 ] && return 0
        [ "$rc" = 2 ] && die "токен рабочий, но доступа к модели нет (см. выше)"
    fi
    [ "$INTERACTIVE" = 1 ] || die "$var не задан или не работает, а спросить некого (нет терминала)"
    for _ in 1 2 3; do
        echo "$hint"
        read -rsp "  $var: " value; echo
        [ -n "$value" ] || continue
        rc=0; $check "$value" || rc=$?
        [ "$rc" = 0 ] && { export "$var=$value"; return 0; }
        [ "$rc" = 2 ] && die "токен рабочий, но доступа к модели нет (см. выше)"
    done
    die "$var так и не прошёл проверку"
}

if [ -n "$KREA2_TRANSFORMER" ] && [ "$KREA2_TRANSFORMER" = "$KREA2_MODELS_DIR/dafK2T_v1.safetensors" ]; then
    NEED_CIVITAI=1
else
    NEED_CIVITAI=0
fi

ask_token HF_TOKEN "Токен Hugging Face (read): huggingface.co/settings/tokens" check_hf
if [ "$NEED_CIVITAI" = 1 ]; then
    ask_token CIVITAI_TOKEN "Ключ CivitAI: civitai.com/user/account -> API Keys" check_civitai
fi

# Пароль на публичную ссылку спрашиваем один раз; пустой ответ тоже запоминаем.
if [ -z "${GRADIO_AUTH_ASKED:-}" ] && [ "$INTERACTIVE" = 1 ] && [ -z "${GRADIO_AUTH:-}" ]; then
    echo "Публичную ссылку сможет открыть любой, у кого она окажется."
    read -rsp "  Пароль для интерфейса (логин krea; Enter — без пароля): " pw; echo
    export GRADIO_AUTH="${pw:+krea:$pw}"
fi
export GRADIO_AUTH_ASKED=1

# Сохраняем всё, чтобы повторный запуск ничего не спрашивал.
umask 077
cat > "$ENV_FILE" <<EOF
# Создано setup.sh. Токены — секреты: файл в .gitignore, никому не отправляйте.
HF_TOKEN="${HF_TOKEN}"
CIVITAI_TOKEN="${CIVITAI_TOKEN:-}"
GRADIO_AUTH="${GRADIO_AUTH:-}"
GRADIO_AUTH_ASKED=1
KREA2_MODEL="${KREA2_MODEL}"
KREA2_TRANSFORMER="${KREA2_TRANSFORMER}"
KREA2_DISTILLED="${KREA2_DISTILLED}"
KREA2_TEXT_ENCODER="${KREA2_TEXT_ENCODER}"
KREA2_PROCESSOR="${KREA2_PROCESSOR}"
KREA2_EDIT_LORA="${KREA2_EDIT_LORA}"
KREA2_MODELS_DIR="${KREA2_MODELS_DIR}"
HF_HOME="${HF_HOME}"
EOF
chmod 600 "$ENV_FILE"
umask 022
echo "  сохранено в $ENV_FILE"
export HF_TOKEN CIVITAI_TOKEN

# ---------- 4. DAF-K2T ----------
say "4/6 трансформер DAF-K2T"
# Файл целый, если его размер совпадает с концом последнего тензора из заголовка.
is_complete() {
    python - "$1" <<'PY'
import json, os, struct, sys
p = sys.argv[1]
try:
    with open(p, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    end = max(v["data_offsets"][1] for v in header.values())
    sys.exit(0 if os.path.getsize(p) == 8 + n + end else 1)
except Exception:
    sys.exit(1)
PY
}

if [ "$NEED_CIVITAI" = 0 ]; then
    echo "свой трансформер: ${KREA2_TRANSFORMER:-штатный из $KREA2_MODEL}"
    [ -z "$KREA2_TRANSFORMER" ] || [ -f "$KREA2_TRANSFORMER" ] || die "нет файла $KREA2_TRANSFORMER"
elif is_complete "$KREA2_TRANSFORMER"; then
    echo "уже скачан: $KREA2_TRANSFORMER"
else
    echo "качаю в $KREA2_TRANSFORMER (12 ГБ; обрыв не страшен — перезапуск докачает)"
    # Authorization при редиректе на CDN curl снимает сам (другой хост).
    curl -L --fail --retry 5 --retry-delay 5 -C - --progress-bar \
         -H "Authorization: Bearer $CIVITAI_TOKEN" \
         -o "$KREA2_TRANSFORMER" \
         "https://civitai.com/api/download/models/$DAF_VERSION_ID"
    is_complete "$KREA2_TRANSFORMER" || die "файл скачался не целиком — запустите setup.sh ещё раз"
    echo "скачан целиком"
fi

# ---------- 5. Hugging Face: энкодер, LoRA, остальное ----------
say "5/6 энкодер, edit-LoRA и компоненты $KREA2_MODEL"
python - <<'PY'
import os
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoProcessor
from krea2_studio.lora import resolve_source

model = os.environ["KREA2_MODEL"]
encoder = os.environ.get("KREA2_TEXT_ENCODER", "")
transformer = os.environ.get("KREA2_TRANSFORMER", "")

# Из репозитория модели берём только то, что не подменяем: свой трансформер и
# свой энкодер from_pretrained качать не станет, и нам не нужно (−33 ГБ).
patterns = ["model_index.json", "scheduler/*", "tokenizer/*", "vae/*"]
if not transformer:
    patterns.append("transformer/*")
if not encoder:
    patterns.append("text_encoder/*")
snapshot_download(model, allow_patterns=patterns)
print(f"  {model}: {', '.join(p.rstrip('/*') for p in patterns)}")

if encoder and not os.path.isdir(encoder):
    snapshot_download(encoder, ignore_patterns=["graphs/*"])
    print(f"  энкодер: {encoder}")

AutoProcessor.from_pretrained(os.environ["KREA2_PROCESSOR"])
print(f"  процессор: {os.environ['KREA2_PROCESSOR']}")

if os.environ.get("KREA2_EDIT_LORA", "").lower() != "off":
    repo, fname = resolve_source()
    if fname:
        hf_hub_download(repo, fname)
    elif not os.path.isfile(repo):
        raise SystemExit(f"нет файла LoRA: {repo}")
    print(f"  edit-LoRA: {repo}{'/' + fname if fname else ''}")
PY

# ---------- 6. запуск ----------
say "6/6 запуск"
echo "модель:   ${KREA2_TRANSFORMER:-$KREA2_MODEL} (distilled=$KREA2_DISTILLED)"
echo "энкодер:  ${KREA2_TEXT_ENCODER:-штатный}"
echo "LoRA:     $KREA2_EDIT_LORA"
echo "offload:  $KREA2_OFFLOAD"
echo "пароль:   $([ -n "${GRADIO_AUTH:-}" ] && echo "да (логин ${GRADIO_AUTH%%:*})" || echo нет)"

if [ "$LAUNCH" = 0 ]; then
    echo -e "\nУстановлено. Запуск: bash setup.sh"
    exit 0
fi

# На vast.ai может работать сервис krea2 с той же моделью: две копии в VRAM не влезут.
if command -v supervisorctl >/dev/null && supervisorctl status krea2 2>/dev/null | grep -q RUNNING; then
    echo "останавливаю сервис krea2 — две копии модели в VRAM не поместятся"
    supervisorctl stop krea2 >/dev/null
fi

echo -e "\nГружу модель (~30 с), затем появится публичная ссылка *.gradio.live."
echo "Остановить — Ctrl+C."
export GRADIO_SHARE=1 KREA2_PRELOAD=1 HOST="${HOST:-0.0.0.0}"
exec python app.py
