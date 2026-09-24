"""Gradio-интерфейс krea2-studio: text2image и edit по референсам.

Запуск:
    export HF_TOKEN=hf_...          # репозитории krea/* gated
    python app.py

Модель грузится ЛЕНИВО — при первом запросе, а не при импорте: иначе HF Space падает
по таймауту старта. Для долгоживущего сервиса лучше KREA2_PRELOAD=1: модель грузится
до открытия порта, и ошибка конфигурации видна в логе сразу, а не по первому клику.
"""

from __future__ import annotations

import os
import random

import gradio as gr
import torch

from krea2_studio.geometry import round_to_multiple

PROCESSOR_ID = os.environ.get("KREA2_PROCESSOR", "Qwen/Qwen3-VL-4B-Instruct")
# Модель, свой трансформер, энкодер и offload читает krea2_studio/loader.py.
# Здесь — только подпись, чтобы в интерфейсе было видно, на чём идёт генерация.
CHECKPOINT = (os.path.basename(os.environ.get("KREA2_TRANSFORMER", ""))
              or os.environ.get("KREA2_MODEL", "krea/Krea-2-Turbo"))
if os.environ.get("KREA2_TEXT_ENCODER"):
    CHECKPOINT += f" + энкодер {os.environ['KREA2_TEXT_ENCODER']}"
DTYPE = torch.bfloat16

# Что лежит во втором слоте вкладки edit.
SLOT_SCENE = "Сцена или вещь"
SLOT_STYLE = "Стиль"

_pipe = None
_processor = None
_lora = None          # откуда взялась edit-LoRA, либо None
_swap = None          # переключатель весов трансформера: DAF-K2T <-> штатный (см. swap.py)

# Режим «Стиль» может идти на штатном Krea-2-Turbo: фотореалистичный файнтюн (DAF-K2T)
# тянет любой стиль обратно к фото. Выбор есть, только если задан свой трансформер.
HAS_CUSTOM = bool(os.environ.get("KREA2_TRANSFORMER"))
STYLE_STOCK = "Штатный Krea 2 Turbo — стиль сильнее"
STYLE_MAIN = f"Основная ({CHECKPOINT.split(' + ')[0]}) — ближе к фото"


def get_pipe():
    """Ленивая загрузка. Turbo — 8 шагов без CFG, Raw — 28 шагов с guidance 4.5."""
    global _pipe, _processor, _lora, _swap
    if _pipe is None:
        from krea2_studio import load_edit_lora, load_pipeline
        from transformers import AutoProcessor

        # Свой трансформер, сменный энкодер, offload — всё по переменным окружения,
        # так же, как в smoke_test.py (см. krea2_studio/loader.py).
        _pipe = load_pipeline(dtype=DTYPE)
        _processor = AutoProcessor.from_pretrained(PROCESSOR_ID)

        # Edit-LoRA: без неё вкладка edit вернёт копию референса (см. krea2_studio/lora.py).
        # Адаптер грузится выключенным и включается только внутри pipe.edit().
        if os.environ.get("KREA2_EDIT_LORA", "").lower() != "off":
            try:
                _lora = load_edit_lora(_pipe)
                print(f"edit-LoRA: {_lora}")
            except Exception as e:
                print(f"ВНИМАНИЕ: edit-LoRA не загрузилась ({type(e).__name__}: {e}).\n"
                      "  Вкладка edit будет копировать референс, не исполняя инструкцию.")

        # После LoRA: переключатель меняет базовые веса под адаптерами.
        from krea2_studio.swap import TransformerSwap
        _swap = TransformerSwap(_pipe, os.environ.get("KREA2_TRANSFORMER") or None)
    return _pipe, _processor


def _seed_to_generator(seed: int):
    if seed is None or seed < 0:
        seed = random.randint(0, 2**31 - 1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.Generator(device).manual_seed(int(seed)), seed


def run_t2i(prompt, negative, height, width, steps, guidance, count, seed,
            progress=gr.Progress(track_tqdm=True)):
    pipe, _ = get_pipe()
    _swap.use("main")
    gen, used = _seed_to_generator(seed)
    images = pipe(
        prompt=prompt,
        negative_prompt=negative or None,
        height=round_to_multiple(height),
        width=round_to_multiple(width),
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        num_images_per_prompt=int(count),
        generator=gen,
    ).images
    return images, f"seed: {used} | модель: {CHECKPOINT}"


def run_edit(prompt, negative, image_a, image_b, height, width, steps, guidance,
             count, seed, ref_boost, ref_boost_scene, fit_mode, auto_pose, grounding,
             grounding_px, system_prompt, slot_mode=SLOT_SCENE, style_keep=0.35,
             style_strength=0.9, style_model=STYLE_STOCK,
             progress=gr.Progress(track_tqdm=True)):
    if image_a is None:
        raise gr.Error("Загрузите хотя бы один референс")

    style_mode = slot_mode == SLOT_STYLE
    if style_mode and image_b is None:
        raise gr.Error("В режиме «Стиль» во второй слот нужна картинка со стилем")

    extra = {}
    if style_mode:
        # Картинка со стилем в латенты не идёт — только в описание стиля словами.
        # Исходник — единственный референс (ослабленный, чтобы стиль проявился)
        # и старт img2img (держит позу, фон и цвета). Размер — по пропорциям
        # исходника с той же площадью, иначе обрезка съест часть кадра.
        images = [image_a]
        ref_boost = float(style_keep)
        extra = dict(style_image=image_b, init_image=image_a, strength=float(style_strength))
        area, ar = float(height) * float(width), image_a.height / image_a.width
        height, width = (area * ar) ** 0.5, (area / ar) ** 0.5
    else:
        # Порядок обучения: сначала сцена, затем субъект. Последний считается основным,
        # на него действует ref_boost — поэтому одиночная картинка идёт субъектом.
        images = [image_a] if image_b is None else [image_b, image_a]

    pipe, processor = get_pipe()
    target = "stock" if style_mode and style_model == STYLE_STOCK else "main"
    swap_s = _swap.use(target)
    gen, used = _seed_to_generator(seed)
    out = pipe.edit(
        prompt=prompt,
        images=images,
        negative_prompt=negative or "",
        height=round_to_multiple(height),
        width=round_to_multiple(width),
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        num_images_per_prompt=int(count),
        generator=gen,
        ref_boost=float(ref_boost),
        ref_boost_scene=float(ref_boost_scene),
        fit_mode=fit_mode,
        auto_pose=bool(auto_pose),
        grounding=bool(grounding),
        grounding_px=int(grounding_px),
        system_prompt=system_prompt or None,
        processor=processor,
        **extra,
    )
    model = CHECKPOINT
    if _swap.available and _swap.current == "stock":
        model = "штатный Krea 2 Turbo" + (" + энкодер " + CHECKPOINT.split(" + энкодер ")[1]
                                          if " + энкодер " in CHECKPOINT else "")
    note = f"seed: {used} | модель: {model} | референсов: {len(images)}"
    if swap_s:
        note += f" | переключение модели: {swap_s:.0f} c"
    if style_mode:
        note += f" | перенос стиля: {round_to_multiple(width)}×{round_to_multiple(height)}"
    note += f" | edit-LoRA: {_lora}" if _lora else " | БЕЗ edit-LoRA: инструкция не исполняется"
    if getattr(pipe, "last_pose_description", None):
        note += f"\n\n**Поза, снятая со сцены:** {pipe.last_pose_description}"
    if getattr(pipe, "last_style_description", None):
        note += f"\n\n**Стиль, снятый со второй картинки:** {pipe.last_style_description}"
    if float(ref_boost) != 1.0 or float(ref_boost_scene) != 1.0:
        note += " | буст включён (медленнее: отключает FlashAttention)"
    return out, note


with gr.Blocks(title="krea2-studio") as demo:
    gr.Markdown("# krea2-studio\nText2image и edit по референсам на чистом diffusers. "
                f"Модель: `{CHECKPOINT}`.")

    with gr.Tab("Text → Image"):
        with gr.Row():
            with gr.Column():
                t_prompt = gr.Textbox(label="Промпт", lines=3)
                t_negative = gr.Textbox(label="Негатив", lines=2)
                with gr.Row():
                    t_h = gr.Slider(256, 2048, 1024, step=16, label="Высота")
                    t_w = gr.Slider(256, 2048, 1024, step=16, label="Ширина")
                with gr.Row():
                    t_steps = gr.Slider(1, 50, 8, step=1, label="Шагов")
                    t_cfg = gr.Slider(0.0, 10.0, 0.0, step=0.1,
                                      label="Guidance (0 = выкл, для Turbo)")
                with gr.Row():
                    t_count = gr.Slider(1, 4, 1, step=1, label="Версий")
                    t_seed = gr.Number(-1, label="Seed (-1 = случайный)", precision=0)
                t_run = gr.Button("Сгенерировать", variant="primary")
            with gr.Column():
                t_out = gr.Gallery(label="Результат", columns=2, height=560)
                t_info = gr.Markdown()
        t_run.click(run_t2i,
                    [t_prompt, t_negative, t_h, t_w, t_steps, t_cfg, t_count, t_seed],
                    [t_out, t_info])

    with gr.Tab("Edit по референсам"):
        with gr.Row():
            with gr.Column():
                gr.Markdown(
                    "Инструкцию исполняет edit-LoRA `krea2-identity-edit` — она грузится "
                    "автоматически при первом запросе. Без неё (`KREA2_EDIT_LORA=off`) "
                    "базовые веса просто скопируют референс.\n\n"
                    "Описывайте **результат**, а не «возьми с первого изображения»: "
                    "ссылки на номера картинок модель разрешает ненадёжно. Позу со "
                    "сцены переносит галочка в «Тонкой настройке».\n\n"
                    "Если нужно просто сменить фон — **грузите только субъекта** и "
                    "опишите фон словами. Референс сцены с человеком превращает задачу "
                    "в двухперсонный монтаж, где лица и одежда смешиваются.\n\n"
                    "Перерисовать первую картинку **в стиле** второй — переключите "
                    "«Вторая картинка — это» на «Стиль»."
                )
                e_prompt = gr.Textbox(label="Инструкция", lines=3,
                                      placeholder="Replace the outfit with a red dress")
                e_negative = gr.Textbox(label="Негатив", lines=2)
                with gr.Row():
                    e_img_a = gr.Image(label="1. СУБЪЕКТ — кого сохраняем (лицо, одежда)",
                                       type="pil")
                    # Второй слот — просто первый референс в последовательности:
                    # туда одинаково идут и фон, и предмет одежды для примерки.
                    e_img_b = gr.Image(label="2. СЦЕНА, ВЕЩЬ или СТИЛЬ — опционально", type="pil")
                # «Стиль»: вторая картинка отдаёт только манеру — всё содержимое, поза
                # и фон остаются с первой. Как сцена она переносила бы и содержимое.
                e_slot = gr.Radio([SLOT_SCENE, SLOT_STYLE], value=SLOT_SCENE,
                                  label="Вторая картинка — это")
                with gr.Group(visible=False) as e_style_box:
                    gr.Markdown("**Перенос стиля.** Первая картинка перерисовывается в стиле "
                                "второй; инструкция необязательна — стиль описывается сам. "
                                "Размер берётся по пропорциям первой картинки.")
                    e_style_keep = gr.Slider(0.1, 1.0, 0.35, step=0.05,
                                             label="Близость к оригиналу: меньше — сильнее стиль, "
                                                   "больше — точнее лицо и детали")
                    # DAF-K2T тянет стиль к фото; штатный Turbo рисует стиль чище.
                    # Переключение весов на месте: ~7 с при смене, VRAM не растёт.
                    e_style_model = gr.Radio([STYLE_STOCK, STYLE_MAIN], value=STYLE_STOCK,
                                             label="Модель для стиля (переключение ~7 с)",
                                             visible=HAS_CUSTOM)
                    e_style_strength = gr.Slider(0.5, 1.0, 0.9, step=0.05,
                                                 label="Свобода перерисовки: 1.0 — с нуля (фон может "
                                                       "перекраситься), 0.85 — держит цвета, но стиль слабее")
                e_slot.change(lambda m: gr.update(visible=m == SLOT_STYLE), e_slot, e_style_box)
                with gr.Row():
                    e_h = gr.Slider(256, 2048, 864, step=16, label="Высота")
                    e_w = gr.Slider(256, 2048, 496, step=16, label="Ширина")
                with gr.Row():
                    # 8 шагов держат композицию, 12 — лицо; 10 посередине.
                    e_steps = gr.Slider(1, 50, 10, step=1, label="Шагов")
                    e_cfg = gr.Slider(0.0, 10.0, 0.0, step=0.1, label="Guidance")
                with gr.Row():
                    e_count = gr.Slider(1, 4, 1, step=1, label="Версий")
                    e_seed = gr.Number(-1, label="Seed (-1 = случайный)", precision=0)
                with gr.Accordion("Тонкая настройка", open=False):
                    e_boost = gr.Slider(0.1, 8.0, 1.0, step=0.05,
                                        label="ref_boost (субъект). 1.0 = выкл, ~4 = сильное сходство")
                    e_boost_s = gr.Slider(0.1, 8.0, 1.0, step=0.05,
                                          label="ref_boost (сцена)")
                    gr.Markdown("Буст насыщается: если модель почти не смотрит на "
                                "референс, `20` даст не больше ~0.5 веса. И он дорог — "
                                "плотная матрица L×L плюс отключение FlashAttention.")
                    # fit — геометрия, под которую обучалась v1.2; crop остался для v1/v1.1.
                    e_fit = gr.Radio(["fit", "crop"], value="fit", label="Подгонка референса")
                    # Поза из сцены сама не переносится — её надо описать словами.
                    # Описание снимается с самого референса, лишней VRAM не стоит.
                    e_pose = gr.Checkbox(False, label="Перенести позу со сцены (авто-описание, +3 c). "
                                                      "Включайте, только если поза нужна")
                    e_ground = gr.Checkbox(True, label="Grounded encode (семантический канал)")
                    e_gpx = gr.Slider(0, 1024, 768, step=64,
                                      label="grounding_px (640–768 в распределении)")
                    e_sys = gr.Textbox(label="Системный промпт (пусто = дефолт обучения)",
                                       lines=2)
                e_run = gr.Button("Редактировать", variant="primary")
            with gr.Column():
                e_out = gr.Gallery(label="Результат", columns=2, height=560)
                e_info = gr.Markdown()
        e_run.click(run_edit,
                    [e_prompt, e_negative, e_img_a, e_img_b, e_h, e_w, e_steps, e_cfg,
                     e_count, e_seed, e_boost, e_boost_s, e_fit, e_pose, e_ground, e_gpx, e_sys,
                     e_slot, e_style_keep, e_style_strength, e_style_model],
                    [e_out, e_info])


if __name__ == "__main__":
    if os.environ.get("KREA2_PRELOAD", "0") == "1":
        get_pipe()
    # Публичная ссылка *.gradio.live открыта всем, у кого она есть: GRADIO_AUTH="логин:пароль"
    # закрывает интерфейс паролем. Пусто — без пароля.
    auth = os.environ.get("GRADIO_AUTH", "")
    auth = tuple(auth.split(":", 1)) if ":" in auth else None
    # За прокси с авторизацией (Caddy на vast.ai) слушаем только 127.0.0.1.
    demo.queue().launch(server_name=os.environ.get("HOST", "0.0.0.0"),
                        server_port=int(os.environ.get("PORT", 7860)),
                        share=os.environ.get("GRADIO_SHARE", "0") == "1",
                        auth=auth)
