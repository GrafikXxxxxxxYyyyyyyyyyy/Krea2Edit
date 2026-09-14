"""Image-grounded энкод инструкции через Qwen3-VL.

Семантический канал edit-режима. В обучении инструкция всегда кодировалась
ВМЕСТЕ с исходным изображением: user-реплика = <vision-токены> + текст.
Стоковый Krea2Pipeline кодирует только текст, поэтому в edit-режиме без этого
модуля отваливается половина рецепта: VAE-токены несут внешность, а этот путь —
семантику сцены («женщина слева», «вывеска на заднем плане»).

Системный префикс здесь БАЙТ В БАЙТ совпадает с `prompt_template_encode_prefix`
из Krea2Pipeline — расхождение здесь ломает раскладку, на которую училась модель.
"""

from __future__ import annotations

import torch
from PIL import Image

# Тот же системный промпт, что и в стоковом текстовом пути Krea2Pipeline.
DEFAULT_SYSTEM_PROMPT = (
    "Describe the image by detailing the color, shape, size, texture, quantity, text, "
    "spatial relationships of the objects and background:"
)

VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"

# LoRA обучалась с джиттером 384-768 px по длинной стороне; 640-768 — в распределении.
DEFAULT_GROUNDING_PX = 768


def build_prefix(system_prompt: str | None = None) -> str:
    """Системная часть — ровно то, что потом отрезается от hidden states."""
    sp = (system_prompt or DEFAULT_SYSTEM_PROMPT).strip()
    return f"<|im_start|>system\n{sp}<|im_end|>\n<|im_start|>user\n"


def downscale_for_grounding(image: Image.Image, grounding_px: int = DEFAULT_GROUNDING_PX) -> Image.Image:
    """Ограничить длинную сторону — VLM не нужен полный размер, а память нужна."""
    if not grounding_px:
        return image.convert("RGB")
    img = image.convert("RGB")
    w, h = img.size
    if max(w, h) <= grounding_px:
        return img
    scale = grounding_px / max(w, h)
    return img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)


@torch.no_grad()
def encode_grounded(
    pipe,
    prompt: str,
    images: list[Image.Image],
    processor,
    grounding_px: int = DEFAULT_GROUNDING_PX,
    system_prompt: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Закодировать инструкцию вместе с изображениями.

    Возвращает ту же пару, что и `pipe.get_text_hidden_states`:
    (hidden_states (B, seq, 12, dim), attention_mask (B, seq)) — то есть результат
    напрямую подставляется в `encoder_hidden_states` трансформера.
    """
    device = pipe._execution_device

    prepared = [downscale_for_grounding(im, grounding_px) for im in images]
    prefix = build_prefix(system_prompt)
    # Vision-блоки идут в порядке обучения: сначала сцена, затем субъект.
    text = prefix + VISION_BLOCK * len(prepared) + (prompt or "") + SUFFIX

    # Процессор сам разворачивает <|image_pad|> в нужное число vision-токенов
    # и считает image_grid_thw — руками это воспроизводить не надо.
    inputs = processor(text=[text], images=prepared, return_tensors="pt").to(device)

    outputs = pipe.text_encoder(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        pixel_values=inputs.pixel_values,
        image_grid_thw=inputs.image_grid_thw,
        output_hidden_states=True,
    )

    hidden_states = torch.stack(
        [outputs.hidden_states[i] for i in pipe.text_encoder_select_layers], dim=2
    )

    # Отрезаем системный префикс — ровно так же, как это делает стоковый путь
    # (там это захардкоженный prompt_template_encode_start_idx=34). Считаем
    # динамически, потому что кастомный системный промпт меняет длину.
    prefix_len = len(processor.tokenizer(prefix, return_tensors="pt").input_ids[0])
    hidden_states = hidden_states[:, prefix_len:]
    attention_mask = inputs.attention_mask[:, prefix_len:].bool()

    return hidden_states, attention_mask
