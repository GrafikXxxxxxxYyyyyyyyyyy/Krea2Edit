"""Edit-LoRA: то, без чего edit-режим не редактирует.

Krea 2 — чистый text2image: в официальном репозитории и в model card нет ни слова
про edit, и базовые веса инструкцию не исполняют. Вся сборка последовательности из
`edit.py` даёт модели референс в контексте — и она послушно его копирует, игнорируя
инструкцию. Навык «прочитать инструкцию и переписать сцену, сохранив личность» живёт
в community-LoRA `krea2-identity-edit`, обученной ровно под эту раскладку
(in-context VAE-токены + grounded encode). Проверено исполнением: без LoRA
«помести этого человека на ночной рынок» возвращает исходный портрет.

Формат ключей у LoRA — ComfyUI/ai-toolkit (`diffusion_model.blocks.N...`), но
diffusers сам разворачивает его в свою раскладку, так что грузится штатным
`load_lora_weights` без конвертера.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

# Веса неофициальные, но это единственная публичная реализация edit для Krea 2;
# ноды, под которые она обучалась, лежат в lbouaraba/comfyui-krea2edit.
DEFAULT_LORA_REPO = "conradlocke/krea2-identity-edit"
DEFAULT_LORA_FILE = "krea2_identity_edit_v1_2.safetensors"

# r64/r128 — те же веса, ужатые SVD; берём их, когда важнее VRAM, чем последний процент.
LOW_VRAM_FILES = {
    "r64": "krea2_identity_edit_v1_2_r64.safetensors",
    "r128": "krea2_identity_edit_v1_2_r128.safetensors",
}

ADAPTER_NAME = "edit"


def resolve_source(source: str | None = None) -> tuple[str, str | None]:
    """(что грузить, имя файла внутри репозитория).

    Принимает: None (дефолт или KREA2_EDIT_LORA), локальный .safetensors,
    ключ 'r64'/'r128' для ужатых вариантов, либо repo_id на Hugging Face.
    """
    source = source or os.environ.get("KREA2_EDIT_LORA") or DEFAULT_LORA_REPO
    if source in LOW_VRAM_FILES:
        return DEFAULT_LORA_REPO, LOW_VRAM_FILES[source]
    if source.endswith(".safetensors"):
        return source, None              # локальный файл или прямой путь в репозитории
    return source, DEFAULT_LORA_FILE


def load_edit_lora(pipe, source: str | None = None, adapter_name: str = ADAPTER_NAME) -> str:
    """Подгрузить edit-LoRA в пайплайн и вернуть то, откуда она взялась.

    Адаптер остаётся ВЫКЛЮЧЕННЫМ: он обучен на edit-раскладке и портит чистый t2i.
    Включать на время edit-вызова — через `edit_lora_active`.
    """
    path, weight_name = resolve_source(source)
    kwargs = {"weight_name": weight_name} if weight_name else {}
    pipe.load_lora_weights(path, adapter_name=adapter_name, **kwargs)
    pipe.disable_lora()
    return f"{path}{'/' + weight_name if weight_name else ''}"


def has_edit_lora(pipe, adapter_name: str = ADAPTER_NAME) -> bool:
    try:
        return adapter_name in (pipe.get_list_adapters().get("transformer") or [])
    except Exception:
        return False


@contextmanager
def edit_lora_active(pipe, adapter_name: str = ADAPTER_NAME, scale: float = 1.0):
    """Включить edit-LoRA на время блока и обязательно выключить после.

    Если адаптер не загружен, контекст ничего не делает — edit-режим остаётся
    работоспособным (референс всё так же влияет), но инструкции модель исполнять
    не будет.
    """
    if not has_edit_lora(pipe, adapter_name):
        yield False
        return
    pipe.set_adapters([adapter_name], adapter_weights=[scale])
    pipe.enable_lora()
    try:
        yield True
    finally:
        pipe.disable_lora()
