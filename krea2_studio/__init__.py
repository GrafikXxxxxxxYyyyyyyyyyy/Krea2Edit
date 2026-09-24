"""krea2-studio — text2image и edit по референсам на чистом diffusers.

Krea2EditPipeline импортируется ЛЕНИВО: он требует diffusers >= 0.40 (где появился
Krea 2), а geometry/attention полезны и без него — например, чтобы разобрать геометрию
на машине без GPU и свежего diffusers.
"""

from .attention import build_ref_bias
from .checkpoint import (comfy_key_to_diffusers, convert_comfy_state_dict, guess_distilled,
                         infer_config, load_transformer)
from .describe import POSE_QUESTION, append_pose, describe_pose
from .encoder import DEFAULT_ENCODER, check_compat, encoder_drift, load_text_encoder
from .geometry import fit_reference, round_to_multiple, PIXELS_PER_TOKEN
from .grounding import DEFAULT_GROUNDING_PX, DEFAULT_SYSTEM_PROMPT
from .lora import DEFAULT_LORA_FILE, DEFAULT_LORA_REPO, has_edit_lora, load_edit_lora

__all__ = [
    "Krea2EditPipeline", "fit_reference", "round_to_multiple", "PIXELS_PER_TOKEN",
    "encode_grounded", "build_ref_bias",
    "DEFAULT_SYSTEM_PROMPT", "DEFAULT_GROUNDING_PX",
    "load_edit_lora", "has_edit_lora", "DEFAULT_LORA_REPO", "DEFAULT_LORA_FILE",
    "describe_pose", "append_pose", "POSE_QUESTION",
    "load_text_encoder", "check_compat", "encoder_drift", "DEFAULT_ENCODER",
    "load_transformer", "convert_comfy_state_dict", "comfy_key_to_diffusers",
    "infer_config", "guess_distilled", "load_pipeline",
]


def __getattr__(name):
    if name == "Krea2EditPipeline":
        try:
            from .edit import Krea2EditPipeline
        except ImportError as e:
            raise ImportError(
                "Krea2EditPipeline требует diffusers с поддержкой Krea 2 (>= 0.40).\n"
                "  pip install 'git+https://github.com/huggingface/diffusers.git'\n"
                f"исходная ошибка: {e}"
            ) from e
        return Krea2EditPipeline
    if name == "load_pipeline":
        from .loader import load_pipeline
        return load_pipeline
    if name == "encode_grounded":
        from .grounding import encode_grounded
        return encode_grounded
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
