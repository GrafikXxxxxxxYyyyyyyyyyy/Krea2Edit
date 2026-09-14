"""ref_boost — аддитивный bias на логиты внимания.

Разобрано в ноутбуке 05: прибавить log(b) к логитам столбцов референса ⟺ умножить
веса внимания на b и ренормализовать (точная формула w' = b·p / (1 + (b-1)·p)).

ВАЖНО про цену. Bias — плотная матрица (1, 1, L, L), которая живёт весь forward и
проходит через все 28 блоков. На 496x864 с одним рефом это ~28 МБ в bf16, на 1328x1328
уже ~390 МБ. Вдобавок явная маска отключает FlashAttention: SDPA сваливается на
math-бэкенд. Поэтому по умолчанию буст выключен (b = 1.0) и bias вообще не строится.
"""

from __future__ import annotations

import math
from contextlib import contextmanager

import torch
import torch.nn.functional as F


def build_ref_bias(
    text_len: int,
    ref_lens: list[int],
    target_len: int,
    boosts: list[float],
    text_mask: torch.Tensor | None = None,
    masks: list[torch.Tensor | None] | None = None,
    ref_grids: list[tuple[int, int]] | None = None,
    device="cpu",
    dtype=torch.float32,
) -> torch.Tensor | None:
    """Bias для последовательности [текст | рефы... | цель].

    Возвращает None, если все бусты равны 1.0 и глушить padding не требуется, —
    это сигнал вызывающему идти быстрым путём без явной маски.
    """
    need_boost = any(b != 1.0 for b in boosts)
    need_pad = text_mask is not None and bool((~text_mask).any())
    if not need_boost and not need_pad:
        return None

    offs = [text_len]
    for n in ref_lens:
        offs.append(offs[-1] + n)
    rows0 = offs[-1]                      # первая строка целевых токенов
    L = rows0 + target_len

    bias = torch.zeros(1, 1, L, L, device=device, dtype=dtype)

    # Паддинг текста глушим для всех строк: эти ключи не должны читаться нигде.
    if need_pad:
        pad_cols = (~text_mask[0]).nonzero(as_tuple=True)[0].to(device)
        bias[:, :, :, pad_cols] = float("-inf")

    for i, b in enumerate(boosts):
        if b == 1.0:
            continue
        off, n = offs[i], ref_lens[i]
        m = masks[i] if masks else None
        if m is not None and ref_grids is not None:
            # Маску региона ужимаем в сетку токенов этого референса.
            gh, gw = ref_grids[i]
            small = F.interpolate(m[None, None].float(), size=(gh, gw), mode="area")[0, 0]
            cols = off + (small.reshape(-1) > 0.5).nonzero(as_tuple=True)[0].to(device)
        else:
            cols = torch.arange(off, off + n, device=device)
        # Буст только на строки цели: вопрос в том, как ЦЕЛЬ смотрит на референс.
        bias[:, :, rows0:, cols] = math.log(max(b, 1e-4))

    return bias


class _BiasHolder:
    """Мутабельный держатель — позволяет менять bias между шагами, не пересоздавая
    процессоры внимания."""

    def __init__(self, bias=None):
        self.bias = bias


class _BoostedProcessor:
    """Обёртка над штатным процессором: подменяет attention_mask нашим bias.

    Штатный forward трансформера строит из encoder_attention_mask key-padding маску
    вида (B,1,1,L) и произвольный additive bias принять не может. Поэтому маску
    подменяем на уровне процессора — наш bias уже включает глушение паддинга.
    """

    def __init__(self, inner, holder: _BiasHolder):
        self.inner = inner
        self.holder = holder

    def __call__(self, attn, hidden_states, attention_mask=None, image_rotary_emb=None, **kw):
        mask = self.holder.bias if self.holder.bias is not None else attention_mask
        return self.inner(attn, hidden_states, attention_mask=mask,
                          image_rotary_emb=image_rotary_emb, **kw)


@contextmanager
def ref_boost_active(transformer, holder: _BiasHolder):
    """Временно подменить процессоры внимания на бустящие.

    NOTE: путь проверен только логически — на игрушечной модели в ноутбуке 05
    эквивалентная механика даёт ожидаемый результат, но подмену процессоров
    у реального Krea2Transformer2DModel надо прогнать на GPU. Если что-то пойдёт
    не так, достаточно не использовать буст: основной edit-путь его не требует.
    """
    original = transformer.attn_processors
    patched = {k: _BoostedProcessor(v, holder) for k, v in original.items()}
    transformer.set_attn_processor(patched)
    try:
        yield
    finally:
        transformer.set_attn_processor(original)
