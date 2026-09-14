"""Загрузка произвольного чекпоинта Krea 2 в формате ComfyUI.

Файнтюны Krea 2 (DAF-K2T и прочие с CivitAI) распространяются одним
`.safetensors` для `ComfyUI/models/diffusion_models/` — это веса только
трансформера, в раскладке имён из `comfy/ldm/krea2/model.py`. `from_pretrained`
такое не берёт: ему нужна структура репозитория с `config.json`.

Здесь имена переводятся в раскладку `Krea2Transformer2DModel`, а остальные
компоненты (VAE, текстовый энкодер, планировщик) берутся из штатного репозитория —
файнтюны трансформера их не трогают.

Маппинг выведен сверкой двух реализаций, а не подбором:

    ComfyUI                          diffusers
    first                            img_in
    tmlp.0 / tmlp.2                  time_embed.linear_1 / linear_2
    tproj.1                          time_mod_proj
    txtmlp.0 / .1 / .3               txt_in.norm / linear_1 / linear_2
    txtfusion                        text_fusion
    blocks.N                         transformer_blocks.N
      .mod.lin                         .scale_shift_table   (6*dim -> (6, dim))
      .prenorm / .postnorm             .norm1 / .norm2
      .attn.wq/wk/wv/gate/wo           .attn.to_q/to_k/to_v/to_gate/to_out.0
      .attn.qknorm.qnorm/.knorm        .attn.norm_q / .norm_k
      .mlp.gate/up/down                .ff.gate/up/down
    last.modulation.lin              final_layer.scale_shift_table
    last.norm / last.linear          final_layer.norm / .linear

Параметр RMSNorm называется `scale` в ComfyUI и `weight` в diffusers; обе
реализации zero-centered (эффективный множитель `1 + w`), так что значения
переносятся как есть.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch

# Префиксы, которыми обрастают веса в разных сборках.
STRIP_PREFIXES = ("model.diffusion_model.", "diffusion_model.", "model.")

_ATTN = {"wq": "to_q", "wk": "to_k", "wv": "to_v", "gate": "to_gate", "wo": "to_out.0"}
_QKNORM = {"qknorm.qnorm": "norm_q", "qknorm.knorm": "norm_k"}


def _strip_prefix(key: str) -> str:
    for p in STRIP_PREFIXES:
        if key.startswith(p):
            return key[len(p):]
    return key


def _convert_attention(rest: str) -> str | None:
    """attn.<что-то> -> diffusers-имя внутри attn."""
    for src, dst in _QKNORM.items():
        if rest.startswith(f"attn.{src}."):
            return f"attn.{dst}." + rest.split(".")[-1]
    m = re.match(r"attn\.(wq|wk|wv|gate|wo)\.(weight|bias)$", rest)
    if m:
        return f"attn.{_ATTN[m.group(1)]}.{m.group(2)}"
    return None


def _convert_block_inner(rest: str) -> str | None:
    """Общая часть SingleStreamBlock и TextFusionBlock."""
    if rest.startswith("prenorm."):
        return "norm1." + rest.split(".")[-1]
    if rest.startswith("postnorm."):
        return "norm2." + rest.split(".")[-1]
    if rest.startswith("attn."):
        return _convert_attention(rest)
    m = re.match(r"mlp\.(gate|up|down)\.(weight|bias)$", rest)
    if m:
        return f"ff.{m.group(1)}.{m.group(2)}"
    return None


def convert_comfy_state_dict(state_dict: dict[str, torch.Tensor],
                             strict: bool = True) -> dict[str, torch.Tensor]:
    """Перевести веса трансформера из раскладки ComfyUI в раскладку diffusers.

    strict=True — падать на ключах, которые не удалось сопоставить (лучше явная
    ошибка, чем молча недогруженная модель).
    """
    out: dict[str, torch.Tensor] = {}
    unmatched: list[str] = []

    for raw_key, value in state_dict.items():
        key = _strip_prefix(raw_key)
        new: str | None = None

        # --- верхний уровень ---
        if key.startswith("first."):
            new = "img_in." + key.split(".")[-1]
        elif key.startswith("tmlp."):
            idx = key.split(".")[1]
            new = {"0": "time_embed.linear_1.", "2": "time_embed.linear_2."}.get(idx)
            new = new + key.split(".")[-1] if new else None
        elif key.startswith("tproj."):
            # Sequential(GELU, Linear) — веса только у элемента 1
            if key.split(".")[1] == "1":
                new = "time_mod_proj." + key.split(".")[-1]
        elif key.startswith("txtmlp."):
            idx = key.split(".")[1]
            if idx == "0":                       # RMSNorm: scale -> weight
                new = "txt_in.norm.weight"
            else:
                head = {"1": "txt_in.linear_1.", "3": "txt_in.linear_2."}.get(idx)
                new = head + key.split(".")[-1] if head else None

        # --- последний слой ---
        elif key.startswith("last."):
            rest = key[len("last."):]
            if rest == "modulation.lin":
                new = "final_layer.scale_shift_table"
            elif rest.startswith("norm."):
                new = "final_layer.norm.weight"
            elif rest.startswith("linear."):
                new = "final_layer.linear." + rest.split(".")[-1]

        # --- блоки трансформера ---
        elif key.startswith("blocks."):
            m = re.match(r"blocks\.(\d+)\.(.+)$", key)
            if m:
                n, rest = m.group(1), m.group(2)
                if rest == "mod.lin":
                    new = f"transformer_blocks.{n}.scale_shift_table"
                else:
                    inner = _convert_block_inner(rest)
                    new = f"transformer_blocks.{n}.{inner}" if inner else None

        # --- стадия слияния текста ---
        elif key.startswith("txtfusion."):
            rest = key[len("txtfusion."):]
            if rest.startswith("projector."):
                new = "text_fusion.projector." + rest.split(".")[-1]
            else:
                m = re.match(r"(layerwise_blocks|refiner_blocks)\.(\d+)\.(.+)$", rest)
                if m:
                    inner = _convert_block_inner(m.group(3))
                    new = f"text_fusion.{m.group(1)}.{m.group(2)}.{inner}" if inner else None

        if new is None:
            unmatched.append(raw_key)
            continue

        # RMSNorm: параметр называется scale в ComfyUI и weight в diffusers
        if new.endswith(".scale"):
            new = new[: -len(".scale")] + ".weight"

        # Таблица модуляции блока хранится плоской (6*dim,), diffusers ждёт (6, dim)
        if new.endswith("scale_shift_table") and value.ndim == 1:
            value = value.reshape(6, -1)

        out[new] = value

    if unmatched and strict:
        head = "\n  ".join(unmatched[:15])
        raise ValueError(
            f"не удалось сопоставить {len(unmatched)} ключей:\n  {head}"
            + ("\n  ..." if len(unmatched) > 15 else "")
            + "\n\nЭто не раскладка ComfyUI для Krea 2 — возможно другая архитектура "
              "или уже diffusers-формат."
        )
    return out


def infer_config(sd: dict[str, torch.Tensor]) -> dict:
    """Восстановить конфиг трансформера из форм тензоров.

    Надёжнее, чем брать конфиг штатного репозитория: если файнтюн менял глубину
    или ширину, мы это увидим, а не упрёмся в несовпадение форм при загрузке.
    """
    n_layers = 1 + max(int(m.group(1))
                       for k in sd if (m := re.match(r"transformer_blocks\.(\d+)\.", k)))
    hidden = sd["img_in.weight"].shape[0]
    in_ch = sd["img_in.weight"].shape[1]
    inter = sd["transformer_blocks.0.ff.gate.weight"].shape[0]
    text_dim = sd["txt_in.norm.weight"].shape[0]
    n_taps = sd["text_fusion.projector.weight"].shape[1]
    q = sd["transformer_blocks.0.attn.to_q.weight"].shape[0]
    kv = sd["transformer_blocks.0.attn.to_k.weight"].shape[0]
    head_dim = sd["transformer_blocks.0.attn.norm_q.weight"].shape[0]

    # RoPE-оси из весов не выводятся (у них нет параметров), но модель требует
    # sum(axes_dims_rope) == attention_head_dim и падает на несовпадении. Для штатного
    # head_dim=128 это (32, 48, 48); для другой ширины делим в той же пропорции 1:1.5:1.5.
    if head_dim == 128:
        axes = (32, 48, 48)
    else:
        t = head_dim // 4
        h_ = (head_dim - t) // 2
        axes = (t, h_, head_dim - t - h_)
        print(f"[checkpoint] нештатный head_dim={head_dim}: axes_dims_rope={axes}")

    return dict(
        axes_dims_rope=axes,
        in_channels=in_ch,
        num_layers=n_layers,
        attention_head_dim=head_dim,
        num_attention_heads=q // head_dim,
        num_key_value_heads=kv // head_dim,
        intermediate_size=inter,
        text_hidden_dim=text_dim,
        num_text_layers=n_taps,
        timestep_embed_dim=sd["time_embed.linear_1.weight"].shape[1],
    )


def load_transformer(path: str | Path, dtype=torch.bfloat16, config_overrides: dict | None = None):
    """Собрать `Krea2Transformer2DModel` из одиночного файла ComfyUI."""
    from safetensors.torch import load_file
    from diffusers import Krea2Transformer2DModel

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"нет файла: {path}")

    raw = load_file(str(path))
    print(f"[checkpoint] {path.name}: {len(raw)} тензоров")

    sd = convert_comfy_state_dict(raw)
    cfg = infer_config(sd)
    cfg.update(config_overrides or {})
    print(f"[checkpoint] конфиг из весов: слоёв {cfg['num_layers']}, "
          f"hidden {cfg['attention_head_dim'] * cfg['num_attention_heads']}, "
          f"тапов {cfg['num_text_layers']}")

    # Веса fp8 встречаются у квантованных сборок; трансформер их не примет.
    sample = next(iter(sd.values()))
    if sample.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        print(f"[checkpoint] веса в {sample.dtype}, привожу к {dtype}")
        sd = {k: v.to(dtype) for k, v in sd.items()}

    model = Krea2Transformer2DModel(**cfg)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"state_dict не сошёлся: не хватает {len(missing)}, лишних {len(unexpected)}.\n"
            f"  не хватает: {missing[:8]}\n  лишние: {unexpected[:8]}")

    return model.to(dtype)


def guess_distilled(name: str) -> bool | None:
    """Угадать по имени, дистиллированный ли чекпоинт. None — не понятно."""
    low = str(name).lower()
    if any(t in low for t in ("turbo", "tdm", "distil", "4step", "8step", "lightning")):
        return True
    if any(t in low for t in ("raw", "base", "midtrain")):
        return False
    return None
