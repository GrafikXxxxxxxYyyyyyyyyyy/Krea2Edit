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

    Возвращает None, если все бусты равны 1.0, — это сигнал вызывающему идти
    быстрым путём со штатной маской и FlashAttention.
    """
    # Без буста bias не нужен вовсе: штатный forward трансформера сам собирает
    # key-padding маску из encoder_attention_mask. Матрицу строим только чтобы поднять
    # референсные столбцы — и тогда уже сами глушим в ней паддинг, потому что нашей
    # маской мы штатную заменяем целиком.
    if not any(b != 1.0 for b in boosts):
        return None
    need_pad = text_mask is not None and bool((~text_mask).any())

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


# Bias живёт на объединённой последовательности [текст | рефы | цель], а её видят
# только основные блоки. text_fusion гоняет внимание по другим осям (по 12 слоям
# энкодера и по токенам текста), и подстановка нашей матрицы туда — гарантированный
# развал формы.
MAIN_BLOCK_PREFIX = "transformer_blocks."


@contextmanager
def ref_boost_active(transformer, holder: _BiasHolder):
    """Временно подменить процессоры внимания основных блоков на бустящие."""
    original = transformer.attn_processors
    patched = {
        k: (_BoostedProcessor(v, holder) if k.startswith(MAIN_BLOCK_PREFIX) else v)
        for k, v in original.items()
    }
    # set_attn_processor вычерпывает словарь через pop — отдаём копии, иначе
    # восстанавливать исходные процессоры будет уже нечем.
    transformer.set_attn_processor(dict(patched))
    try:
        yield
    finally:
        transformer.set_attn_processor(dict(original))
