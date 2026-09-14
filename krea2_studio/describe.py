"""Автоописание позы со второго референса.

Зачем. Поза из референса сцены сама по себе НЕ переносится — проверено: тот же
кадр, тот же сид, в промпте ни слова о позе — фон меняется, а поза остаётся
исходной. Копирование позы не входит в натренированные навыки edit-LoRA; работает
только словесное описание в инструкции. Прописывать его руками под каждую картинку
нереально, поэтому описание снимаем с самого референса.

Чем. Тем же Qwen3-VL, что уже стоит текстовым энкодером пайплайна. У чекпоинта
`tie_word_embeddings=True`, то есть lm_head — это матрица эмбеддингов, и она уже
в памяти: генератор поверх неё стоит РОВНО НОЛЬ дополнительной VRAM (измерено:
34.2 ГБ до и после) и около 3 секунд на описание.
"""

from __future__ import annotations

import torch
from PIL import Image

# Просим ровно геометрию тела. Без этого ограничения VLM сползает на одежду и фон,
# а они должны прийти из своих каналов — платье из субъекта, фон из сцены.
POSE_QUESTION = (
    "Describe ONLY the body pose of the person: the position of the torso, arms, "
    "hands, legs and head, and the camera angle. One sentence, no mention of "
    "clothing, background, colors or identity."
)


def _captioner(pipe):
    """Обёртка с lm_head поверх уже загруженного энкодера. Строится один раз."""
    lm = getattr(pipe, "_krea2_captioner", None)
    if lm is not None:
        return lm

    from accelerate import init_empty_weights
    from transformers import Qwen3VLForConditionalGeneration

    # Скелет на meta-устройстве: настоящие веса не аллоцируются вообще.
    with init_empty_weights():
        lm = Qwen3VLForConditionalGeneration(pipe.text_encoder.config)
    lm.model = pipe.text_encoder
    lm.tie_weights()          # lm_head <- embed_tokens, обе уже на GPU
    lm.eval()
    pipe._krea2_captioner = lm
    return lm


@torch.no_grad()
def describe_pose(pipe, image: Image.Image, processor,
                  question: str = POSE_QUESTION, max_new_tokens: int = 90) -> str:
    """Одно предложение про позу человека на картинке."""
    lm = _captioner(pipe)
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image.convert("RGB")},
        {"type": "text", "text": question},
    ]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(pipe._execution_device)
    inputs["pixel_values"] = inputs["pixel_values"].to(pipe.text_encoder.dtype)

    # Жадная генерация: описание должно быть воспроизводимым от прогона к прогону,
    # иначе один и тот же сид даёт разную позу.
    ids = lm.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = processor.batch_decode(
        ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    return text.strip()


def append_pose(prompt: str, pose: str) -> str:
    """Приклеить описание позы к инструкции."""
    if not pose:
        return prompt
    return f"{prompt.rstrip().rstrip('.')}. Pose: {pose}"
