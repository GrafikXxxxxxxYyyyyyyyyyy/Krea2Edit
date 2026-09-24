"""Сборка пайплайна по переменным окружения — общая для app.py и smoke_test.py.

Раньше подмену трансформера умел только app.py, и диагностика проверяла штатные веса,
даже когда задан KREA2_TRANSFORMER. Теперь оба входа собирают пайплайн одинаково.

    KREA2_MODEL          базовый репозиторий: VAE, энкодер, планировщик (и трансформер,
                         если свой не задан)
    KREA2_TRANSFORMER    одиночный .safetensors в раскладке ComfyUI (см. checkpoint.py)
    KREA2_DISTILLED      1/0 — тип своего чекпоинта; пусто — угадать по имени файла
    KREA2_TEXT_ENCODER   сменный энкодер, только Qwen3-VL-4B (см. encoder.py)
    KREA2_OFFLOAD        1 — CPU-offload вместо целиком на GPU
"""

from __future__ import annotations

import os

import torch

from .checkpoint import guess_distilled, load_transformer
from .encoder import check_compat, load_text_encoder


def load_pipeline(dtype=torch.bfloat16):
    """Собрать Krea2EditPipeline и разместить его на устройстве."""
    from .edit import Krea2EditPipeline

    model_id = os.environ.get("KREA2_MODEL", "krea/Krea-2-Turbo")
    transformer_file = os.environ.get("KREA2_TRANSFORMER", "")
    distilled_env = os.environ.get("KREA2_DISTILLED", "")
    encoder_id = os.environ.get("KREA2_TEXT_ENCODER", "")

    extra = {}
    if transformer_file:
        print(f"трансформер: {transformer_file} (вместо штатного)")
        extra["transformer"] = load_transformer(transformer_file, dtype=dtype)
    if encoder_id:
        print(f"текстовый энкодер: {encoder_id} (вместо штатного)")
        extra["text_encoder"] = load_text_encoder(encoder_id, dtype=dtype)

    # Переданные компоненты from_pretrained не качает — штатный трансформер
    # при своём чекпоинте с диска не тянется.
    pipe = Krea2EditPipeline.from_pretrained(model_id, dtype=dtype, **extra)

    if encoder_id:
        # Форму проверяем до первой генерации, а не посреди денойз-лупа.
        check_compat(pipe.text_encoder, pipe.transformer, pipe.text_encoder_select_layers)

    if transformer_file:
        # is_distilled определяет сдвиг расписания (mu) и дефолты шагов/guidance.
        # Ошибиться тут дороже, чем спросить: Turbo на расписании Raw даёт мыло.
        if distilled_env:
            distilled = distilled_env == "1"
            src = "KREA2_DISTILLED"
        else:
            guessed = guess_distilled(transformer_file)
            distilled = pipe.config.is_distilled if guessed is None else guessed
            src = "имя файла" if guessed is not None else "репозиторий KREA2_MODEL"
            if guessed is None:
                print("ВНИМАНИЕ: по имени файла не понять, дистиллированный ли чекпоинт.\n"
                      "  Если результат мыльный или пережжённый — задай KREA2_DISTILLED=1 или 0.")
        pipe.register_to_config(is_distilled=distilled)
        print(f"is_distilled={distilled} (источник: {src})")

    # Offload — ВМЕСТО переноса на GPU: сначала .to("cuda"), потом offload
    # падает по памяти раньше, чем offload успевает помочь.
    if os.environ.get("KREA2_OFFLOAD", "0") == "1":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    pipe.vae.enable_tiling()
    return pipe
