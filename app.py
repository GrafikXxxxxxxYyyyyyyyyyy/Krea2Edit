"""Gradio-интерфейс krea2-studio: text2image и edit по референсам.

Запуск:
    export HF_TOKEN=hf_...          # репозитории krea/* gated
    python app.py

Модель грузится ЛЕНИВО — при первом запросе, а не при импорте: иначе HF Space падает
по таймауту старта, а на vast.ai не увидишь ошибку конфигурации до первой генерации.
"""

from __future__ import annotations

import os
import random

import gradio as gr
import torch

from krea2_studio.geometry import round_to_multiple

MODEL_ID = os.environ.get("KREA2_MODEL", "krea/Krea-2-Turbo")
PROCESSOR_ID = os.environ.get("KREA2_PROCESSOR", "Qwen/Qwen3-VL-4B-Instruct")
DTYPE = torch.bfloat16

_pipe = None
_processor = None


def get_pipe():
    """Ленивая загрузка. Turbo — 8 шагов без CFG, Raw — 28 шагов с guidance 4.5."""
    global _pipe, _processor
    if _pipe is None:
        from krea2_studio import Krea2EditPipeline
        from transformers import AutoProcessor

        _pipe = Krea2EditPipeline.from_pretrained(MODEL_ID, dtype=DTYPE)
        _pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        # Экономия VRAM: пригодится на картах меньше 40 ГБ.
        if os.environ.get("KREA2_OFFLOAD", "0") == "1":
            _pipe.enable_model_cpu_offload()
        _pipe.vae.enable_tiling()
        _processor = AutoProcessor.from_pretrained(PROCESSOR_ID)
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
    return images, f"seed: {used}"


def run_edit(prompt, negative, image_a, image_b, height, width, steps, guidance,
             count, seed, ref_boost, ref_boost_scene, fit_mode, grounding,
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
        grounding=bool(grounding),
        grounding_px=int(grounding_px),
        system_prompt=system_prompt or None,
        processor=processor,
    )
    note = f"seed: {used} | референсов: {len(images)}"
    if float(ref_boost) != 1.0 or float(ref_boost_scene) != 1.0:
        note += " | буст включён (медленнее: отключает FlashAttention)"
    return out, note


with gr.Blocks(title="krea2-studio") as demo:
    gr.Markdown("# krea2-studio\nText2image и edit по референсам на чистом diffusers.")

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
                e_prompt = gr.Textbox(label="Инструкция", lines=3,
                                      placeholder="Replace the outfit with a red dress")
                e_negative = gr.Textbox(label="Негатив", lines=2)
                with gr.Row():
                    e_img_a = gr.Image(label="Референс (субъект)", type="pil")
                    e_img_b = gr.Image(label="Референс 2 — сцена (опционально)", type="pil")
                with gr.Row():
                    e_h = gr.Slider(256, 2048, 864, step=16, label="Высота")
                    e_w = gr.Slider(256, 2048, 496, step=16, label="Ширина")
                with gr.Row():
                    e_steps = gr.Slider(1, 50, 8, step=1, label="Шагов")
                    e_cfg = gr.Slider(0.0, 10.0, 0.0, step=0.1, label="Guidance")
                with gr.Row():
                    e_count = gr.Slider(1, 4, 1, step=1, label="Версий")
                    e_seed = gr.Number(-1, label="Seed (-1 = случайный)", precision=0)
                with gr.Accordion("Тонкая настройка", open=False):
                    e_boost = gr.Slider(0.1, 8.0, 1.0, step=0.05,
                                        label="ref_boost (субъект). 1.0 = выкл")
                    e_boost_s = gr.Slider(0.1, 8.0, 1.0, step=0.05,
                                          label="ref_boost (сцена)")
                    gr.Markdown("Буст насыщается: если модель почти не смотрит на "
                                "референс, `20` даст не больше ~0.5 веса. И он дорог — "
                                "плотная матрица L×L плюс отключение FlashAttention.")
                    e_fit = gr.Radio(["crop", "fit"], value="crop", label="Подгонка референса")
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
                     e_count, e_seed, e_boost, e_boost_s, e_fit, e_ground, e_gpx, e_sys],
                    [e_out, e_info])


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0",
                        server_port=int(os.environ.get("PORT", 7860)),
                        share=os.environ.get("GRADIO_SHARE", "0") == "1")
