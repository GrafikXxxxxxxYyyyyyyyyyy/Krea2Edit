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


def comfy_key_to_diffusers(raw_key: str) -> str | None:
    """Имя тензора ComfyUI -> имя в `Krea2Transformer2DModel`. None — не сопоставилось.

    Работает только с именами, без значений: так маппинг можно прогнать по заголовку
    safetensors, не читая сами веса.
    """
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

    # RMSNorm: параметр называется scale в ComfyUI и weight в diffusers
    if new is not None and new.endswith(".scale"):
        new = new[: -len(".scale")] + ".weight"
    return new


def _fix_value(new_key: str, value: torch.Tensor) -> torch.Tensor:
    """Таблица модуляции блока хранится плоской (6*dim,), diffusers ждёт (6, dim)."""
    if new_key.endswith("scale_shift_table") and value.ndim == 1:
        return value.reshape(6, -1)
    return value


def _raise_unmatched(unmatched: list[str]) -> None:
    head = "\n  ".join(unmatched[:15])
    raise ValueError(
        f"не удалось сопоставить {len(unmatched)} ключей:\n  {head}"
        + ("\n  ..." if len(unmatched) > 15 else "")
        + "\n\nЭто не раскладка ComfyUI для Krea 2 — возможно другая архитектура "
          "или уже diffusers-формат."
    )


def convert_comfy_state_dict(state_dict: dict[str, torch.Tensor],
                             strict: bool = True) -> dict[str, torch.Tensor]:
    """Перевести веса трансформера из раскладки ComfyUI в раскладку diffusers.

    strict=True — падать на ключах, которые не удалось сопоставить (лучше явная
    ошибка, чем молча недогруженная модель).
    """
    out: dict[str, torch.Tensor] = {}
    unmatched: list[str] = []

    for raw_key, value in state_dict.items():
        new = comfy_key_to_diffusers(raw_key)
        if new is None:
            unmatched.append(raw_key)
            continue
        out[new] = _fix_value(new, value)

    if unmatched and strict:
        _raise_unmatched(unmatched)
    return out


def infer_config(sd: dict[str, torch.Tensor]) -> dict:
    """Восстановить конфиг трансформера из форм тензоров.

    Надёжнее, чем брать конфиг штатного репозитория: если файнтюн менял глубину
    или ширину, мы это увидим, а не упрёмся в несовпадение форм при загрузке.
    Хватает форм — значения не читаются, так что годятся и тензоры на meta.
    """
    def count(pattern):
        idx = [int(m.group(1)) for k in sd if (m := re.match(pattern, k))]
        return 1 + max(idx) if idx else 0

    n_layers = count(r"transformer_blocks\.(\d+)\.")
    hidden = sd["img_in.weight"].shape[0]
    in_ch = sd["img_in.weight"].shape[1]
    inter = sd["transformer_blocks.0.ff.gate.weight"].shape[0]
    text_dim = sd["txt_in.norm.weight"].shape[0]
    n_taps = sd["text_fusion.projector.weight"].shape[1]
    kv = sd["transformer_blocks.0.attn.to_k.weight"].shape[0]
    head_dim = sd["transformer_blocks.0.attn.norm_q.weight"].shape[0]

    # У text_fusion свои головы и своя ширина MLP: из дефолтов diffusers их брать
    # нельзя — нештатный энкодер или урезанный фьюжн развалятся на конструкторе.
    tf = "text_fusion.layerwise_blocks.0"
    text_head_dim = sd[f"{tf}.attn.norm_q.weight"].shape[0]
    text_kv = sd[f"{tf}.attn.to_k.weight"].shape[0]
    text_inter = sd[f"{tf}.ff.gate.weight"].shape[0]

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
        num_attention_heads=hidden // head_dim,
        num_key_value_heads=kv // head_dim,
        intermediate_size=inter,
        text_hidden_dim=text_dim,
        num_text_layers=n_taps,
        text_num_attention_heads=text_dim // text_head_dim,
        text_num_key_value_heads=text_kv // text_head_dim,
        text_intermediate_size=text_inter,
        num_layerwise_text_blocks=count(r"text_fusion\.layerwise_blocks\.(\d+)\."),
        num_refiner_text_blocks=count(r"text_fusion\.refiner_blocks\.(\d+)\."),
        timestep_embed_dim=sd["time_embed.linear_1.weight"].shape[1],
    )


def load_transformer(path: str | Path, dtype=torch.bfloat16, config_overrides: dict | None = None):
    """Собрать `Krea2Transformer2DModel` из одиночного файла ComfyUI.

    Про память. Наивный путь — load_file целиком, затем модель в fp32 и
    load_state_dict — держит одновременно сырые веса, их копию в bf16 и fp32-модель:
    на 12B это ~85 ГБ RAM. Здесь модель собирается на meta-устройстве, а тензоры
    читаются из файла по одному, сразу приводятся к целевому типу и встают в модель
    без копии (assign=True). Пик — примерно размер модели в целевом dtype.
    """
    from accelerate import init_empty_weights
    from safetensors import safe_open
    from diffusers import Krea2Transformer2DModel

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"нет файла: {path}")

    with safe_open(str(path), framework="pt") as f:
        raw_keys = list(f.keys())
        print(f"[checkpoint] {path.name}: {len(raw_keys)} тензоров")

        names = {k: comfy_key_to_diffusers(k) for k in raw_keys}
        unmatched = [k for k, v in names.items() if v is None]
        if unmatched:
            _raise_unmatched(unmatched)

        # Конфиг — по формам из заголовка, до чтения весов.
        shapes = {names[k]: _fix_value(names[k], torch.empty(f.get_slice(k).get_shape(),
                                                              device="meta"))
                  for k in raw_keys}
        cfg = infer_config(shapes)
        cfg.update(config_overrides or {})
        print(f"[checkpoint] конфиг из весов: слоёв {cfg['num_layers']}, "
              f"hidden {cfg['attention_head_dim'] * cfg['num_attention_heads']}, "
              f"тапов {cfg['num_text_layers']}")

        with init_empty_weights():
            model = Krea2Transformer2DModel(**cfg)

        # Нормы держим в fp32 — ровно так их оставляет from_pretrained
        # (_keep_in_fp32_modules), иначе файнтюн считался бы не так, как штатный.
        keep_fp32 = model._keep_in_fp32_modules or []
        sd, src_dtypes = {}, {}
        for k in raw_keys:
            new = names[k]
            t = f.get_tensor(k)
            src_dtypes[t.dtype] = src_dtypes.get(t.dtype, 0) + 1
            want = torch.float32 if any(m in new.split(".") for m in keep_fp32) else dtype
            sd[new] = _fix_value(new, t.to(want))

    print("[checkpoint] типы в файле: "
          + ", ".join(f"{str(d).replace('torch.', '')} x{n}" for d, n in src_dtypes.items())
          + f" -> {str(dtype).replace('torch.', '')} (нормы fp32)")

    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    if missing or unexpected:
        raise ValueError(
            f"state_dict не сошёлся: не хватает {len(missing)}, лишних {len(unexpected)}.\n"
            f"  не хватает: {missing[:8]}\n  лишние: {unexpected[:8]}")

    # Метка источника: smoke_test сверяет по ней, что пайплайн собран на этом файле.
    model._krea2_source = str(path)
    return model.eval()


def guess_distilled(name: str) -> bool | None:
    """Угадать по имени, дистиллированный ли чекпоинт. None — не понятно."""
    low = str(name).lower()
    if any(t in low for t in ("turbo", "tdm", "distil", "4step", "8step", "lightning")):
        return True
    if any(t in low for t in ("raw", "base", "midtrain")):
        return False
    return None
