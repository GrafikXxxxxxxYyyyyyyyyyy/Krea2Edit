"""Геометрия подгонки референса под целевую сетку.

Правила выведены и измерены в ноутбуке 06:

  1. center-crop к целевому AR, ПОТОМ resize — наивный interpolate растянул
     контрольный круг на 75%;
  2. подгонять в пиксельном пространстве: ресайз латентов теряет ~44% высоких частот;
  3. снап на /16 (vae_scale_factor * patch_size), и обязательно с подрезкой источника
     под сетку — иначе содержимое сплющивается до 15 px, и рассогласование выпадает
     ровно на границу референсного блока (тот самый удвоенный шов);
  4. при неполном покрытии центрировать ДРОБНЫМ офсетом — округление стоит 8 px.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PIXELS_PER_TOKEN = 16          # vae_scale_factor(8) * patch_size(2)

# Если после вписывания промах по обеим осям меньше 8%, режим fit вырождается в crop:
# поля в 1-2 токена не безобидны — краевые столбцы цели остаются без соответствия
# в референсе, и модель дублирует туда ближайшее содержимое.
CROP_TOLERANCE = 0.08


def round_to_multiple(value: int, multiple: int = PIXELS_PER_TOKEN) -> int:
    """Округлить вверх до кратного: целевое разрешение обязано быть кратно 16."""
    return int(multiple * ((int(value) + multiple - 1) // multiple))


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """PIL -> float32 (1, 3, H, W) в [0, 1]."""
    arr = np.asarray(image.convert("RGB")).copy()   # copy: иначе torch ругается на non-writable
    return torch.from_numpy(arr).float().div(255.0).permute(2, 0, 1).unsqueeze(0)


def _crop_then_resize(px: torch.Tensor, th: int, tw: int) -> torch.Tensor:
    """Center-crop до целевого AR, затем resize. Пропорции не искажаются."""
    ih, iw = px.shape[-2:]
    s = max(th / ih, tw / iw)
    ch, cw = min(ih, int(round(th / s))), min(iw, int(round(tw / s)))
    y0, x0 = (ih - ch) // 2, (iw - cw) // 2
    px = px[..., y0:y0 + ch, x0:x0 + cw]
    return F.interpolate(px, size=(th, tw), mode="bicubic", antialias=True).clamp(0, 1)


def _fit_with_crop_to_grid(px: torch.Tensor, th: int, tw: int,
                           snap: int = PIXELS_PER_TOKEN) -> torch.Tensor:
    """Вписать целиком, попав на сетку БЕЗ сплющивания.

    Ключевой момент: размер снапится вниз, а источник подрезается так, чтобы при том же
    масштабе лечь на сетку точно. Без подрезки масштаб по осям расходится, и шов
    проступает на границе референсного блока.
    """
    ih, iw = px.shape[-2:]
    sc = min(th / ih, tw / iw)
    nh = min(max(snap, int(ih * sc) // snap * snap), max(snap, th // snap * snap))
    nw = min(max(snap, int(iw * sc) // snap * snap), max(snap, tw // snap * snap))

    ch = min(ih, max(1, int(round(nh / sc))))
    cw = min(iw, max(1, int(round(nw / sc))))
    y0, x0 = (ih - ch) // 2, (iw - cw) // 2
    px = px[..., y0:y0 + ch, x0:x0 + cw]
    return F.interpolate(px, size=(nh, nw), mode="bicubic", antialias=True).clamp(0, 1)


def fit_reference(image: Image.Image, target_h: int, target_w: int, mode: str = "crop",
                  snap: int = PIXELS_PER_TOKEN, tol: float = CROP_TOLERANCE):
    """Привести референс к целевой сетке.

    Возвращает (пиксели (1,3,H,W) в [0,1], (gh, gw), (off_h, off_w)), где gh/gw —
    сетка токенов референса, а офсеты — его дробное центрирование внутри целевой сетки.

    mode="crop": обрезать до AR цели, заполнить сетку целиком (офсеты нулевые);
    mode="fit" : вписать целиком; при близком AR автоматически вырождается в crop.
    """
    px = image if isinstance(image, torch.Tensor) else pil_to_tensor(image)
    ih, iw = px.shape[-2:]

    if mode == "fit":
        sc = min(target_h / ih, target_w / iw)
        covers = (ih * sc >= target_h * (1 - tol)) and (iw * sc >= target_w * (1 - tol))
        if not covers:
            out = _fit_with_crop_to_grid(px, target_h, target_w, snap)
            gh, gw = out.shape[-2] // snap, out.shape[-1] // snap
            th_, tw_ = target_h // snap, target_w // snap
            return out, (gh, gw), (max(0.0, (th_ - gh) / 2), max(0.0, (tw_ - gw) / 2))
    elif mode != "crop":
        raise ValueError(f"неизвестный режим подгонки: {mode!r} (ожидалось 'crop' или 'fit')")

    out = _crop_then_resize(px, target_h, target_w)
    return out, (target_h // snap, target_w // snap), (0.0, 0.0)
