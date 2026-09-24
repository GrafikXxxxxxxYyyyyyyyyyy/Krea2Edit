"""Пошаговая проверка на реальных весах.

Запускать ПЕРЕД app.py: скрипт идёт от простого к сложному и на каждом шаге
печатает, что именно проверяется и что делать, если упало. Так ошибка
локализуется сразу, а не всплывает где-то внутри Gradio.

    python smoke_test.py                 # всё, 496x864, 8 шагов
    python smoke_test.py --steps 4       # быстрее
    python smoke_test.py --only 1,2,3    # только часть шагов

Картинки складываются в ./smoke_out/.
"""

from __future__ import annotations

import argparse
import os
import time
import traceback
from pathlib import Path

OUT = Path("smoke_out")
results: list[tuple[str, str, str]] = []      # (шаг, статус, комментарий)


def step(num: int, title: str, hint: str):
    """Декоратор: ловит исключение, печатает подсказку, не роняет остальные шаги."""
    def deco(fn):
        def wrapper(ctx, *a, **kw):
            print(f"\n{'='*70}\n[{num}] {title}\n{'='*70}")
            t0 = time.time()
            try:
                note = fn(ctx, *a, **kw) or ""
                dt = time.time() - t0
                print(f"  OK  ({dt:.1f} c) {note}")
                results.append((f"{num}. {title}", "OK", f"{dt:.1f} c {note}"))
                return True
            except Exception as e:
                dt = time.time() - t0
                print(f"  ПРОВАЛ ({dt:.1f} c): {type(e).__name__}: {e}")
                print(f"  ЧТО ДЕЛАТЬ: {hint}")
                traceback.print_exc()
                results.append((f"{num}. {title}", "ПРОВАЛ", f"{type(e).__name__}: {e}"))
                return False
        return wrapper
    return deco


@step(0, "Окружение", "Проверь requirements.txt: нужен diffusers из git и transformers>=4.57")
def check_env(ctx):
    import torch, transformers, diffusers
    print(f"  torch        {torch.__version__}")
    print(f"  diffusers    {diffusers.__version__}")
    print(f"  transformers {transformers.__version__}")
    print(f"  CUDA         {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"  GPU          {p.name}, {p.total_memory/1024**3:.1f} ГБ")
        ctx["vram_gb"] = p.total_memory / 1024**3
    else:
        print("  ВНИМАНИЕ: CUDA нет — на CPU это будет невыносимо медленно")
        ctx["vram_gb"] = 0
    from diffusers import Krea2Pipeline           # noqa: F401  — главная проверка
    return "Krea2Pipeline доступен"


@step(1, "Загрузка модели и edit-LoRA",
      "401 -> прими условия на странице модели и задай HF_TOKEN. OOM -> KREA2_OFFLOAD=1. "
      "Если не встала LoRA -> pip install peft")
def load_model(ctx):
    import torch
    from krea2_studio import load_edit_lora, load_pipeline
    from transformers import AutoProcessor

    model_id = os.environ.get("KREA2_MODEL", "krea/Krea-2-Turbo")
    print(f"  модель: {model_id}  (первый запуск качает ~30 ГБ)")
    # Тот же путь сборки, что и в app.py: KREA2_TRANSFORMER, KREA2_DISTILLED,
    # KREA2_TEXT_ENCODER и KREA2_OFFLOAD учитываются здесь же.
    pipe = load_pipeline(dtype=torch.bfloat16)

    ctx["pipe"] = pipe
    ctx["processor"] = AutoProcessor.from_pretrained(
        os.environ.get("KREA2_PROCESSOR", "Qwen/Qwen3-VL-4B-Instruct"))

    n = sum(p.numel() for p in pipe.transformer.parameters())
    print(f"  трансформер: {n/1e9:.2f}B параметров (оценка в ноутбуке 07 была 12.16B)")
    print(f"  is_distilled: {pipe.config.is_distilled}")

    # Без edit-LoRA шаги 3-8 отработают, но редактированием это не будет:
    # базовые веса скопируют референс и проигнорируют инструкцию.
    ctx["lora"] = None
    if os.environ.get("KREA2_EDIT_LORA", "").lower() == "off":
        print("  edit-LoRA: пропущена (KREA2_EDIT_LORA=off)")
    else:
        ctx["lora"] = load_edit_lora(pipe)
        print(f"  edit-LoRA: {ctx['lora']}")

    if torch.cuda.is_available():
        print(f"  занято VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} ГБ")
    return f"{n/1e9:.2f}B" + ("" if ctx["lora"] else ", без LoRA")


@step(2, "t2i — базовая проверка", "Если сломан t2i, edit отлаживать бессмысленно: "
                                   "дело в загрузке весов или в самом diffusers")
def test_t2i(ctx, h, w, steps):
    import torch
    pipe = ctx["pipe"]
    g = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(0)
    img = pipe(prompt="a red fox in the snow, photo", height=h, width=w,
               num_inference_steps=steps,
               guidance_scale=0.0 if pipe.config.is_distilled else 4.5,
               generator=g).images[0]
    OUT.mkdir(exist_ok=True)
    img.save(OUT / "01_t2i.png")
    ctx["ref_image"] = img                      # используем как референс дальше
    return f"-> {OUT/'01_t2i.png'}"


@step(3, "edit без grounding и без буста", "Шум на выходе -> ошибка в position_ids или срезе. "
                                           "Результат идентичен t2i -> референс не доехал до attention")
def test_edit_plain(ctx, h, w, steps):
    import torch
    pipe = ctx["pipe"]
    g = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(0)
    out = pipe.edit(prompt="make the fox blue", images=[ctx["ref_image"]],
                    height=h, width=w, num_inference_steps=steps,
                    grounding=False, ref_boost=1.0, generator=g)
    out[0].save(OUT / "02_edit_plain.png")
    return f"-> {OUT/'02_edit_plain.png'}"


@step(4, "Референс реально влияет", "Если картинки совпали — референс игнорируется: "
                                    "проверь position_ids (ось кадра) и порядок конкатенации")
def test_ref_matters(ctx, h, w, steps):
    import torch
    from PIL import Image
    import numpy as np
    pipe = ctx["pipe"]

    # заведомо другой референс: ровная заливка
    other = Image.fromarray(np.full((h, w, 3), (20, 200, 60), dtype=np.uint8))
    g = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(0)
    out = pipe.edit(prompt="make the fox blue", images=[other],
                    height=h, width=w, num_inference_steps=steps,
                    grounding=False, ref_boost=1.0, generator=g)
    out[0].save(OUT / "03_edit_other_ref.png")

    a = np.asarray(Image.open(OUT / "02_edit_plain.png"), dtype=np.float32)
    b = np.asarray(out[0], dtype=np.float32)
    diff = np.abs(a - b).mean()
    print(f"  средняя разница при смене референса: {diff:.2f} (из 255)")
    if diff < 1.0:
        raise RuntimeError(f"референс почти не влияет (разница {diff:.2f}) — см. подсказку")
    return f"разница {diff:.1f}"


@step(5, "grounded encode (vision-токены в условии)",
      "Если падает в text_encoder — Qwen3VLModel ждёт других аргументов; "
      "смотри сигнатуру forward у своей версии transformers (с 5.x нужен mm_token_type_ids)")
def test_grounding(ctx, h, w, steps):
    import torch
    from krea2_studio.grounding import encode_grounded
    pipe = ctx["pipe"]

    emb, mask = encode_grounded(pipe, "make the fox blue", [ctx["ref_image"]],
                                ctx["processor"], grounding_px=768)
    n = emb.shape[1]
    print(f"  длина условия: {n} токенов")
    print(f"  ~350 -> vision-токены В условии (так и должно быть)")
    print(f"  ~13  -> они отброшены, нужно менять prefix_len")
    ctx["grounded_len"] = n

    g = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(0)
    out = pipe.edit(prompt="make the fox blue", images=[ctx["ref_image"]],
                    height=h, width=w, num_inference_steps=steps,
                    grounding=True, processor=ctx["processor"], generator=g)
    out[0].save(OUT / "04_edit_grounded.png")
    return f"условие {n} токенов -> {OUT/'04_edit_grounded.png'}"


@step(6, "ref_boost (подмена процессоров внимания)",
      "Если падает — set_attn_processor не принял обёртку. Не блокирует: "
      "буст опционален, просто не используй его")
def test_boost(ctx, h, w, steps):
    import torch
    pipe = ctx["pipe"]
    g = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(0)
    out = pipe.edit(prompt="make the fox blue", images=[ctx["ref_image"]],
                    height=h, width=w, num_inference_steps=steps,
                    grounding=False, ref_boost=3.0, generator=g)
    out[0].save(OUT / "05_edit_boost3.png")
    return f"-> {OUT/'05_edit_boost3.png'}"


@step(7, "Геометрия: референс с чужим aspect ratio",
      "Растянутое изображение -> перепутан порядок crop/resize в geometry.py")
def test_geometry(ctx, h, w, steps):
    import torch
    pipe = ctx["pipe"]
    wide = ctx["ref_image"].resize((1200, 400))      # заведомо другой AR
    for mode in ["crop", "fit"]:
        g = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(0)
        out = pipe.edit(prompt="make the fox blue", images=[wide],
                        height=h, width=w, num_inference_steps=steps,
                        grounding=False, fit_mode=mode, generator=g)
        out[0].save(OUT / f"06_geom_{mode}.png")
    return f"-> {OUT}/06_geom_crop.png, 06_geom_fit.png"


@step(8, "edit-LoRA доезжает до forward",
      "Если разницы нет — адаптер не включился: проверь pip install peft и что "
      "load_edit_lora отработал на шаге 1")
def test_lora_effect(ctx, h, w, steps):
    import torch
    import numpy as np
    pipe = ctx["pipe"]
    if not ctx.get("lora"):
        raise RuntimeError("edit-LoRA не загружена (шаг 1) — сравнивать не с чем")

    # lora_scale=0 обнуляет вклад адаптера, всё остальное в проходе идентично:
    # любая разница в результате — это ровно LoRA.
    outs = {}
    for tag, scale in [("off", 0.0), ("on", 1.0)]:
        g = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(0)
        out = pipe.edit(prompt="turn the fox into a blue fox", images=[ctx["ref_image"]],
                        height=h, width=w, num_inference_steps=steps,
                        grounding=True, processor=ctx["processor"], fit_mode="fit",
                        lora_scale=scale, generator=g)
        out[0].save(OUT / f"07_lora_{tag}.png")
        outs[tag] = np.asarray(out[0], dtype=np.float32)

    diff = np.abs(outs["on"] - outs["off"]).mean()
    print(f"  разница LoRA on/off: {diff:.2f} (из 255)")
    if diff < 1.0:
        raise RuntimeError(f"LoRA ничего не меняет (разница {diff:.2f}) — см. подсказку")
    return f"разница {diff:.1f} -> {OUT}/07_lora_off.png, 07_lora_on.png"


@step(9, "auto_pose: описание позы со второго референса",
      "Если падает на generate — проверь, что у чекпоинта tie_word_embeddings=True "
      "и transformers знает Qwen3VLForConditionalGeneration")
def test_auto_pose(ctx, h, w, steps):
    import torch
    import numpy as np
    from PIL import Image
    pipe = ctx["pipe"]

    # Сцена с ЯВНОЙ позой: лис стоит, а тут человек с поднятыми руками.
    before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    scene = pipe(prompt="photo of a person standing with both arms raised high above the head",
                 height=h, width=w, num_inference_steps=steps,
                 guidance_scale=0.0 if pipe.config.is_distilled else 4.5).images[0]

    g = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(0)
    out = pipe.edit(prompt="create a photo of this fox on a city street",
                    images=[scene, ctx["ref_image"]], height=h, width=w,
                    num_inference_steps=steps, grounding=True, processor=ctx["processor"],
                    fit_mode="fit", auto_pose=True, generator=g)
    out[0].save(OUT / "08_auto_pose.png")

    pose = pipe.last_pose_description
    print(f"  описание: {pose}")
    if not pose or len(pose) < 20:
        raise RuntimeError("описание позы пустое или подозрительно короткое")

    # Генератор поверх связанных эмбеддингов не должен стоить заметной VRAM.
    if torch.cuda.is_available():
        grown = (torch.cuda.memory_allocated() - before) / 1024**3
        print(f"  прирост VRAM за счёт генератора: {grown:+.2f} ГБ")
    return f"{len(pose)} симв. -> {OUT/'08_auto_pose.png'}"


@step(10, "подмена текстового энкодера",
      "Ошибка формы -> это не Qwen3-VL-4B (нужен hidden 2560 / 36 слоёв). "
      "OOM -> шаг держит второй энкодер (~8 ГБ), запусти его отдельно: --only 0,1,10")
def test_encoder_swap(ctx, h, w, steps):
    from krea2_studio import DEFAULT_ENCODER, check_compat, encoder_drift, load_text_encoder
    import torch
    pipe = ctx["pipe"]

    # Веса энкодера у Krea сверены с upstream потензорно: все 713 совпадают. Шумового
    # пола нет — второй экземпляр тех же весов даёт побитово те же hidden states, —
    # значит и upstream обязан совпасть бит в бит. Иначе сломан путь подмены (например,
    # частоты RoPE огрублены до bf16 — см. encoder_drift).
    other = load_text_encoder(DEFAULT_ENCODER, dtype=pipe.text_encoder.dtype)
    check_compat(other, pipe.transformer, pipe.text_encoder_select_layers)
    d = encoder_drift(pipe, other)
    del other
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"  upstream Qwen3-VL-4B-Instruct: mean={d['mean']}, identical={d['identical']}")
    print(f"  косинус по слоям: {d['per_layer']}")
    if not d["identical"]:
        raise RuntimeError(
            f"upstream разошёлся со штатным (mean {d['mean']}) — "
            "либо сломан путь подмены, либо веса всё-таки не те")
    return "upstream совпал со штатным бит в бит"


@step(11, "конвертер чекпоинтов ComfyUI",
      "Маппинг имён разошёлся с текущим diffusers — смотри krea2_studio/checkpoint.py")
def test_checkpoint_mapping(ctx, h, w, steps):
    """Самопроверка конвертера: синтетический ComfyUI-файл -> Krea2Transformer2DModel.

    Не требует ни настоящего файла, ни GPU — ловит расхождение с diffusers до того,
    как ты скачаешь двенадцать гигабайт файнтюна. Проверяется весь путь загрузки,
    а не только имена: крошечная эталонная модель сохраняется в раскладке ComfyUI
    (большие матрицы в fp8, как у DAF-K2T), грузится обратно через load_transformer
    и сверяется потензорно вместе с конфигом.

    Если задан KREA2_TRANSFORMER, дополнительно проверяется, что пайплайн из шага 1
    собран именно на этом файле, и на нём генерируется картинка.
    """
    import os
    import tempfile
    import torch
    from diffusers import Krea2Transformer2DModel
    from safetensors.torch import save_file
    from krea2_studio.checkpoint import comfy_key_to_diffusers, infer_config, load_transformer

    # Нештатные размеры везде, где конфиг выводится из форм: дефолт diffusers
    # не должен совпасть с ответом случайно.
    cfg0 = dict(in_channels=16, num_layers=2, attention_head_dim=16, num_attention_heads=4,
                num_key_value_heads=2, intermediate_size=128, timestep_embed_dim=8,
                text_hidden_dim=32, num_text_layers=12, text_num_attention_heads=2,
                text_num_key_value_heads=1, text_intermediate_size=48,
                num_layerwise_text_blocks=2, num_refiner_text_blocks=3,
                axes_dims_rope=(4, 6, 6))
    torch.manual_seed(0)
    ref = {k: torch.randn_like(v) for k, v in Krea2Transformer2DModel(**cfg0).state_dict().items()}

    # Имена — так, как их пишет ComfyUI (comfy/ldm/krea2/model.py).
    def blk(prefix):
        names = [f"{prefix}.prenorm.scale", f"{prefix}.postnorm.scale",
                 f"{prefix}.attn.qknorm.qnorm.scale", f"{prefix}.attn.qknorm.knorm.scale"]
        names += [f"{prefix}.attn.{n}.weight" for n in ("wq", "wk", "wv", "gate", "wo")]
        names += [f"{prefix}.mlp.{n}.weight" for n in ("gate", "up", "down")]
        return names

    names = ["first.weight", "first.bias", "tmlp.0.weight", "tmlp.0.bias",
             "tmlp.2.weight", "tmlp.2.bias", "tproj.1.weight", "tproj.1.bias",
             "txtmlp.0.scale", "txtmlp.1.weight", "txtmlp.1.bias",
             "txtmlp.3.weight", "txtmlp.3.bias", "txtfusion.projector.weight",
             "last.norm.scale", "last.linear.weight", "last.linear.bias", "last.modulation.lin"]
    for i in range(cfg0["num_layers"]):
        names += blk(f"blocks.{i}") + [f"blocks.{i}.mod.lin"]
    for grp, n in (("layerwise_blocks", cfg0["num_layerwise_text_blocks"]),
                   ("refiner_blocks", cfg0["num_refiner_text_blocks"])):
        for i in range(n):
            names += blk(f"txtfusion.{grp}.{i}")

    comfy, expected = {}, {}
    for raw in names:
        new = comfy_key_to_diffusers(raw)
        if new is None or new not in ref:
            raise AssertionError(f"{raw} -> {new}: такого имени в diffusers нет")
        v = ref[new]
        if new.startswith("transformer_blocks.") and new.endswith("scale_shift_table"):
            v = v.flatten()                           # ComfyUI хранит (6*dim,)
        if v.ndim == 2 and "norm" not in new:
            v = v.to(torch.float8_e4m3fn)             # как у fp8-сборок с CivitAI
        comfy["model.diffusion_model." + raw] = v.contiguous()
        expected[new] = v
    missing = set(ref) - set(expected)
    if missing:
        raise AssertionError(f"маппинг не покрывает {len(missing)} имён diffusers: "
                             f"{sorted(missing)[:6]}")

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "synthetic_comfy.safetensors")
        save_file(comfy, path)
        model = load_transformer(path, dtype=torch.bfloat16)

    got_cfg = {k: model.config[k] for k in cfg0}
    got_cfg["axes_dims_rope"] = tuple(got_cfg["axes_dims_rope"])
    if got_cfg != cfg0:
        diff = {k: (cfg0[k], got_cfg[k]) for k in cfg0 if cfg0[k] != got_cfg[k]}
        raise AssertionError(f"конфиг из форм разошёлся (ждали, получили): {diff}")

    sd = model.state_dict()
    for k, v in expected.items():
        want_dtype = torch.float32 if "norm" in k.split(".")[-2] else torch.bfloat16
        if sd[k].dtype != want_dtype:
            raise AssertionError(f"{k}: dtype {sd[k].dtype}, ждали {want_dtype}")
        if sd[k].device.type == "meta":
            raise AssertionError(f"{k}: остался на meta — вес не загрузился")
        want = v.float().reshape(sd[k].shape).to(want_dtype)
        if not torch.equal(sd[k], want):
            raise AssertionError(f"{k}: значения не совпали после загрузки")
    print(f"  синтетический файл: {len(comfy)} тензоров, конфиг и значения сошлись, "
          "fp8 -> bf16, нормы fp32")

    path = os.environ.get("KREA2_TRANSFORMER", "")
    if not path:
        return "синтетика OK (KREA2_TRANSFORMER не задан — реальный файл не проверялся)"
    if "pipe" not in ctx:
        return "синтетика OK (реальный файл проверяется вместе с шагом 1: --only 1,11)"

    from krea2_studio.checkpoint import guess_distilled
    pipe = ctx["pipe"]
    src = getattr(pipe.transformer, "_krea2_source", None)
    if src != str(path):
        raise AssertionError(f"пайплайн собран не на {path} (трансформер: {src or 'штатный'})")
    print(f"  трансформер из файла: {path}")
    print(f"  is_distilled={pipe.config.is_distilled} (по имени: {guess_distilled(path)})")
    g = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(0)
    img = pipe(prompt="a red fox in the snow, photo", height=h, width=w,
               num_inference_steps=steps,
               guidance_scale=0.0 if pipe.config.is_distilled else 4.5,
               generator=g).images[0]
    img.save(OUT / "11_custom_checkpoint.png")
    return f"файл загружен -> {OUT/'11_custom_checkpoint.png'}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=int, default=864)
    ap.add_argument("--width", type=int, default=496)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--only", type=str, default="", help="например 0,1,2")
    args = ap.parse_args()

    only = {int(x) for x in args.only.split(",") if x.strip()} if args.only else None
    ctx: dict = {}
    h, w, s = args.height, args.width, args.steps

    plan = [
        (0, check_env, ()),
        (1, load_model, ()),
        (2, test_t2i, (h, w, s)),
        (3, test_edit_plain, (h, w, s)),
        (4, test_ref_matters, (h, w, s)),
        (5, test_grounding, (h, w, s)),
        (6, test_boost, (h, w, s)),
        (7, test_geometry, (h, w, s)),
        (8, test_lora_effect, (h, w, s)),
        (9, test_auto_pose, (h, w, s)),
        (10, test_encoder_swap, (h, w, s)),
        (11, test_checkpoint_mapping, (h, w, s)),
    ]

    print(f"разрешение {h}x{w}, шагов {s}")
    for num, fn, fnargs in plan:
        if only is not None and num not in only:
            continue
        ok = fn(ctx, *fnargs)
        if not ok and num <= 1:
            print("\nбазовый шаг провален — дальше идти нет смысла")
            break

    print(f"\n{'='*70}\nИТОГ\n{'='*70}")
    for name, status, note in results:
        mark = "+" if status == "OK" else "!"
        print(f" {mark} {name:<48} {status:<8} {note}")

    if "grounded_len" in ctx:
        n = ctx["grounded_len"]
        print(f"\nvision-токены: условие вышло {n} токенов -> "
              f"{'они в условии (как и задумано)' if n > 100 else 'ОТБРОШЕНЫ, надо править prefix_len'}")

    failed = [r for r in results if r[1] != "OK"]
    print(f"\nпровалов: {len(failed)} из {len(results)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
