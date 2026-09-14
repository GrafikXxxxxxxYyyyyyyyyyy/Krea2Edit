"""Замена текстового энкодера.

Энкодер у Krea 2 — **немодифицированный** `Qwen/Qwen3-VL-4B-Instruct`: сверены все
713 тензоров из `text_encoder/model.safetensors`, расхождение ноль. Отличается только
раскладка ключей (в репозитории Krea они без префикса `model.`), и transformers
разбирает обе сами.

Отсюда два следствия. Первое: любой дериватив Qwen3-VL-**4B** встаёт на место как есть
— форма совпадает, словарь тот же. Второе, менее приятное: `text_fusion` трансформера
обучался против hidden states ИМЕННО стоковых весов, так что любой файнтюн — это
смещение распределения на входе диффузии. Насколько большое, показывает `encoder_drift`:
считайте дрейф ДО того, как жечь часы GPU на выяснение, стало ли лучше.

Размер менять нельзя: 8B даёт hidden 4096 против требуемых 2560 — `txt_in` и
`text_fusion.projector` просто не сойдутся по форме.
"""

from __future__ import annotations

import torch

DEFAULT_ENCODER = "Qwen/Qwen3-VL-4B-Instruct"


def load_text_encoder(source: str = DEFAULT_ENCODER, dtype=torch.bfloat16, **kwargs):
    """Загрузить сменный энкодер. Принимает repo_id или локальный путь."""
    from transformers import Qwen3VLModel
    return Qwen3VLModel.from_pretrained(source, dtype=dtype, **kwargs)


def check_compat(text_encoder, transformer, select_layers) -> None:
    """Проверить контракт до первой генерации, а не посреди денойз-лупа."""
    cfg = text_encoder.config.text_config
    want_dim = transformer.config.text_hidden_dim
    want_taps = transformer.config.num_text_layers

    if cfg.hidden_size != want_dim:
        raise ValueError(
            f"энкодер отдаёт hidden_size={cfg.hidden_size}, а трансформеру нужно "
            f"{want_dim}. Это другой размер модели — подходит только Qwen3-VL-4B."
        )
    if len(select_layers) != want_taps:
        raise ValueError(
            f"снимается {len(select_layers)} слоёв, трансформер ждёт {want_taps}")
    if max(select_layers) >= cfg.num_hidden_layers + 1:
        raise ValueError(
            f"снимается слой {max(select_layers)}, а в энкодере их "
            f"{cfg.num_hidden_layers}")


@torch.no_grad()
def encoder_drift(pipe, other_encoder, prompt: str = "a photograph of a city street",
                  max_sequence_length: int = 512) -> dict:
    """Насколько hidden states чужого энкодера расходятся со стоковыми.

    Возвращает косинусную близость по каждому из 12 снимаемых слоёв и в среднем.
    Чем ниже, тем дальше вход диффузии от того, на чём её учили.

    ВАЖНО про шумовой пол: ровно 1.0 не бывает даже при побитово одинаковых весах.
    Второй экземпляр тех же весов уже даёт ~0.99995 — у него свои адреса активаций,
    и в bf16 это меняет выбор ядер cuBLAS. Измерено: второй экземпляр весов Krea и
    upstream Qwen3-VL-4B-Instruct дают одинаковые 0.99995. Так что сравнивать дрейф
    чужого файнтюна надо не с единицей, а с этим полом.
    """
    import torch.nn.functional as F

    original = pipe.text_encoder
    try:
        a, mask = pipe.get_text_hidden_states(prompt, max_sequence_length, pipe._execution_device)
        pipe.text_encoder = other_encoder.to(pipe._execution_device, dtype=original.dtype)
        b, _ = pipe.get_text_hidden_states(prompt, max_sequence_length, pipe._execution_device)
    finally:
        pipe.text_encoder = original

    keep = mask[0]                                   # паддинг в счёт не берём
    a, b = a[0][keep].float(), b[0][keep].float()    # (токены, 12, dim)
    per_layer = [F.cosine_similarity(a[:, i], b[:, i], dim=-1).mean().item()
                 for i in range(a.shape[1])]
    return {
        "per_layer": [round(v, 6) for v in per_layer],
        "mean": round(sum(per_layer) / len(per_layer), 6),
        "worst": round(min(per_layer), 6),
        "identical": bool(torch.equal(a, b)),
    }
