"""Подмена весов трансформера на месте: два чекпоинта, одна копия в VRAM.

Зачем. DAF-K2T — фотореалистичный файнтюн и сопротивляется нефото-стилям: режим
«Стиль» на нём даёт полуреализм, а на штатном Krea-2-Turbo — чистое аниме (замерено
на одной паре, тот же рецепт). Держать оба трансформера нельзя: 24 + 24 ГБ плюс
энкодер не влезают в 48 ГБ.

Как. Архитектура у файнтюна та же, поэтому достаточно переписать значения
параметров: тензоры читаются из файла по одному и копируются в уже лежащие на GPU
параметры (`param.copy_`). Память не растёт, адаптеры PEFT (edit-LoRA) остаются на
месте — меняются только базовые веса под ними.
"""

from __future__ import annotations

import glob
import os
import time
from pathlib import Path

import torch

from .checkpoint import _fix_value, comfy_key_to_diffusers

STOCK = "stock"
MAIN = "main"


def _stock_files(model_id: str) -> list[str]:
    """Шарды transformer/ штатного репозитория (скачиваются при необходимости)."""
    from huggingface_hub import snapshot_download
    root = snapshot_download(model_id, allow_patterns=["transformer/*"])
    files = sorted(glob.glob(os.path.join(root, "transformer", "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"в {model_id} нет transformer/*.safetensors")
    return files


def _iter_tensors(source: str, model_id: str):
    """(имя в diffusers, тензор) из одиночного файла ComfyUI или из штатного репозитория."""
    from safetensors import safe_open
    if source == STOCK:
        for path in _stock_files(model_id):
            with safe_open(path, framework="pt") as f:
                for k in f.keys():
                    yield k, f.get_tensor(k)
    else:
        with safe_open(source, framework="pt") as f:
            for k in f.keys():
                name = comfy_key_to_diffusers(k)
                if name is None:
                    raise ValueError(f"{source}: не удалось сопоставить ключ {k}")
                yield name, _fix_value(name, f.get_tensor(k))


class TransformerSwap:
    """Держит на GPU один трансформер и переключает его веса между чекпоинтами.

        swap = TransformerSwap(pipe, main_file="/path/DAF-K2T.safetensors")
        swap.use("stock")   # штатный Krea-2-Turbo
        swap.use("main")    # обратно файнтюн
    """

    def __init__(self, pipe, main_file: str | None, model_id: str | None = None):
        self.pipe = pipe
        self.model_id = model_id or os.environ.get("KREA2_MODEL", "krea/Krea-2-Turbo")
        self.sources = {MAIN: main_file or STOCK, STOCK: STOCK}
        self.current = MAIN
        # Базовые параметры без адаптеров: у PEFT-обёрток вес лежит в .base_layer.
        self.params = {}
        for n, p in pipe.transformer.named_parameters():
            if "lora_" in n:
                continue
            self.params[n.replace(".base_layer", "")] = p

    @property
    def available(self) -> bool:
        """Есть что переключать: свой трансформер задан и отличается от штатного."""
        return self.sources[MAIN] != STOCK

    @torch.no_grad()
    def use(self, name: str) -> float:
        """Переключиться на чекпоинт; вернуть потраченные секунды (0 — уже на нём)."""
        if name == self.current or self.sources[name] == self.sources[self.current]:
            self.current = name
            return 0.0
        t0 = time.time()
        seen = set()
        for key, tensor in _iter_tensors(self.sources[name], self.model_id):
            p = self.params.get(key)
            if p is None:
                raise KeyError(f"в трансформере нет параметра {key}")
            if tuple(p.shape) != tuple(tensor.shape):
                raise ValueError(f"{key}: форма {tuple(tensor.shape)} против {tuple(p.shape)}")
            p.copy_(tensor.to(p.device, non_blocking=True))   # dtype параметра сохраняется
            seen.add(key)
        missing = set(self.params) - seen
        if missing:
            # Половина весов от одной модели, половина от другой — хуже падения.
            raise RuntimeError(f"не записано {len(missing)} параметров: {sorted(missing)[:5]}")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.current = name
        dt = time.time() - t0
        print(f"[swap] трансформер -> {name} ({Path(str(self.sources[name])).name}) за {dt:.1f} c")
        return dt
