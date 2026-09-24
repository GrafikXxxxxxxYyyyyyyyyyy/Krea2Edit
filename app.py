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

_pipe = None
_processor = None
_lora = None          # откуда взялась edit-LoRA, либо None


def get_pipe():
    """Ленивая загрузка. Turbo — 8 шагов без CFG, Raw — 28 шагов с guidance 4.5."""
    global _pipe, _processor, _lora
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
    return _pipe, _processor


def _seed_to_generator(seed: int):
    if seed is None or seed < 0:
        seed = random.randint(0, 2**31 - 1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.Generator(device).manual_seed(int(seed)), seed


def run_t2i(prompt, negative, height, width, steps, guidance, count, seed,
            progress=gr.Progress(track_tqdm=True)):
    pipe, _ = get_pipe()
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
             grounding_px, system_prompt, progress=gr.Progress(track_tqdm=True)):
    if image_a is None:
        raise gr.Error("Загрузите хотя бы один референс")

    # Порядок обучения: сначала сцена, затем субъект. Последний считается основным,
    # на него действует ref_boost — поэтому одиночная картинка идёт субъектом.
    images = [image_a] if image_b is None else [image_b, image_a]

    pipe, processor = get_pipe()
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
    )
    note = f"seed: {used} | модель: {CHECKPOINT} | референсов: {len(images)}"
    note += f" | edit-LoRA: {_lora}" if _lora else " | БЕЗ edit-LoRA: инструкция не исполняется"
    if getattr(pipe, "last_pose_description", None):
        note += f"\n\n**Поза, снятая со сцены:** {pipe.last_pose_description}"
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
                    "в двухперсонный монтаж, где лица и одежда смешиваются."
                )
                e_prompt = gr.Textbox(label="Инструкция", lines=3,
                                      placeholder="Replace the outfit with a red dress")
                e_negative = gr.Textbox(label="Негатив", lines=2)
                with gr.Row():
                    e_img_a = gr.Image(label="1. СУБЪЕКТ — кого сохраняем (лицо, одежда)",
                                       type="pil")
                    # Второй слот — просто первый референс в последовательности:
                    # туда одинаково идут и фон, и предмет одежды для примерки.
                    e_img_b = gr.Image(label="2. СЦЕНА или ВЕЩЬ — опционально", type="pil")
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
                     e_count, e_seed, e_boost, e_boost_s, e_fit, e_pose, e_ground, e_gpx, e_sys],
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
