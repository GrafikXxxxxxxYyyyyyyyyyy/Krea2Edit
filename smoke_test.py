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


@step(1, "Загрузка модели", "401 -> прими условия на странице модели и задай HF_TOKEN. "
                            "OOM -> KREA2_OFFLOAD=1")
def load_model(ctx):
    import torch
    from krea2_studio import Krea2EditPipeline
    from transformers import AutoProcessor

    model_id = os.environ.get("KREA2_MODEL", "krea/Krea-2-Turbo")
    print(f"  модель: {model_id}  (первый запуск качает ~30 ГБ)")
    pipe = Krea2EditPipeline.from_pretrained(model_id, dtype=torch.bfloat16)

    if os.environ.get("KREA2_OFFLOAD", "0") == "1":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    pipe.vae.enable_tiling()

    ctx["pipe"] = pipe
    ctx["processor"] = AutoProcessor.from_pretrained(
        os.environ.get("KREA2_PROCESSOR", "Qwen/Qwen3-VL-4B-Instruct"))

    n = sum(p.numel() for p in pipe.transformer.parameters())
    print(f"  трансформер: {n/1e9:.2f}B параметров (оценка в ноутбуке 07 была 12.16B)")
    print(f"  is_distilled: {pipe.config.is_distilled}")
    if torch.cuda.is_available():
        print(f"  занято VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} ГБ")
    return f"{n/1e9:.2f}B"


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


@step(5, "grounded encode + ОТКРЫТЫЙ ВОПРОС про vision-токены",
      "Если падает в text_encoder — Qwen3VLModel ждёт других аргументов; "
      "смотри сигнатуру forward у своей версии transformers")
def test_grounding(ctx, h, w, steps):
    import torch
    from krea2_studio.grounding import encode_grounded
    pipe = ctx["pipe"]

    emb, mask = encode_grounded(pipe, "make the fox blue", [ctx["ref_image"]],
                                ctx["processor"], grounding_px=768)
    n = emb.shape[1]
    print(f"  длина условия: {n} токенов")
    print(f"  ~350 -> vision-токены В условии (наша реализация, ноутбук 04)")
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
