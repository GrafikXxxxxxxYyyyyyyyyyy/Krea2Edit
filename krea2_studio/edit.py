"""Krea2EditPipeline — edit-режим поверх стокового Krea2Pipeline.

Механика разобрана в ноутбуках 02-06. Кратко:

    последовательность:  [ текст | реф 1 | реф 2 | цель ]
    позиции (t,h,w):     (0,0,0)  (1,h,w) (2,h,w) (0,h,w)
    выход:               только целевые токены, срез с конца

Текст в hidden_states НЕ попадает — его приклеивает сам трансформер. А в position_ids
попадает. Рассинхрон здесь — типовая ошибка.
"""

from __future__ import annotations

import warnings
from contextlib import ExitStack

import torch
from PIL import Image

from diffusers import Krea2Pipeline
from diffusers.pipelines.krea2.pipeline_krea2 import calculate_shift, retrieve_timesteps

from .attention import _BiasHolder, build_ref_bias, ref_boost_active
from .describe import append_pose, describe_pose
from .geometry import fit_reference, round_to_multiple
from .grounding import encode_grounded
from .lora import edit_lora_active, has_edit_lora


class Krea2EditPipeline(Krea2Pipeline):
    """Krea2Pipeline + edit по референсам.

    Наследование даёт нам _pack_latents / _unpack_latents / prepare_latents /
    get_text_hidden_states и весь набор компонентов. Стоковый __call__ (text2image)
    продолжает работать как раньше.
    """

    # ---------- референсы ----------

    @staticmethod
    def frame_ids(gh: int, gw: int, frame: int = 0,
                  off_h: float = 0.0, off_w: float = 0.0, device="cpu") -> torch.Tensor:
        """Координаты (t, h, w) для сетки токенов. Офсеты дробные — RoPE непрерывен."""
        ids = torch.zeros(gh, gw, 3, device=device)
        ids[..., 0] = frame
        ids[..., 1] = (torch.arange(gh, device=device).float() + off_h)[:, None]
        ids[..., 2] = (torch.arange(gw, device=device).float() + off_w)[None, :]
        return ids.reshape(gh * gw, 3)

    def prepare_edit_position_ids(self, text_len, ref_specs, gh, gw, device):
        """[текст в нуле | рефы с кадрами 1..N | цель с кадром 0]"""
        parts = [torch.zeros(text_len, 3, device=device)]
        for i, (rgh, rgw, off_h, off_w) in enumerate(ref_specs):
            parts.append(self.frame_ids(rgh, rgw, i + 1, off_h, off_w, device))
        parts.append(self.frame_ids(gh, gw, 0, device=device))
        return torch.cat(parts, dim=0)

    @torch.no_grad()
    def encode_reference(self, image: Image.Image, height: int, width: int,
                         mode: str = "crop", dtype=None):
        """Картинка -> упакованный чистый латент + описание его сетки.

        Подгонка идёт в ПИКСЕЛЬНОМ пространстве (ноутбук 06: ресайз латентов теряет
        ~44% высоких частот), нормализация — поканальная.
        """
        device = self._execution_device
        dtype = dtype or self.vae.dtype

        px, (gh, gw), (off_h, off_w) = fit_reference(image, height, width, mode=mode)
        px = (px * 2.0 - 1.0).to(device=device, dtype=dtype)     # [0,1] -> [-1,1]

        # AutoencoderKLQwenImage работает с 5D (B, C, T, H, W); для картинки T=1.
        lat = self.vae.encode(px.unsqueeze(2)).latent_dist.mode()

        mean = torch.tensor(self.vae.config.latents_mean, device=device, dtype=lat.dtype)
        std = torch.tensor(self.vae.config.latents_std, device=device, dtype=lat.dtype)
        lat = (lat - mean.view(1, -1, 1, 1, 1)) / std.view(1, -1, 1, 1, 1)

        lat = lat.squeeze(2)                                     # (B, z, h, w)
        z = lat.shape[1]
        packed = self._pack_latents(lat, lat.shape[0], z, lat.shape[-2], lat.shape[-1])
        return packed, (gh, gw, off_h, off_w)

    # ---------- один шаг ----------

    def edit_velocity(self, latents, refs_packed, prompt_embeds, prompt_mask,
                      timestep, position_ids):
        """Вызов трансформера в edit-режиме. Возвращает скорость ТОЛЬКО для цели."""
        hidden = torch.cat(list(refs_packed) + [latents], dim=1)
        out = self.transformer(
            hidden_states=hidden,
            encoder_hidden_states=prompt_embeds,
            timestep=timestep,
            position_ids=position_ids,
            encoder_attention_mask=prompt_mask,
            attention_kwargs=self.attention_kwargs,
            return_dict=False,
        )[0]
        return out[:, -latents.shape[1]:]          # референсы отрезаем с начала

    # ---------- полный проход ----------

    @torch.no_grad()
    def edit(
        self,
        prompt: str,
        images: list[Image.Image],
        negative_prompt: str = "",
        height: int = 864,
        width: int = 496,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        num_images_per_prompt: int = 1,
        generator=None,
        ref_boost: float = 1.0,
        ref_boost_scene: float = 1.0,
        boost_mask=None,
        fit_mode: str = "crop",
        lora_scale: float = 1.0,
        auto_pose: bool = False,
        grounding: bool = True,
        grounding_px: int = 768,
        system_prompt: str | None = None,
        processor=None,
        max_sequence_length: int = 512,
        output_type: str = "pil",
    ):
        """Редактирование по одному-двум референсам.

        images: порядок обучения — сначала сцена, затем субъект. Последний считается
        основным, на него действует ref_boost (остальным достаётся ref_boost_scene).
        """
        if not images:
            raise ValueError("edit() требует хотя бы один референс; для t2i зови сам пайплайн")

        # Поза из референса сцены сама не переносится (см. describe.py) — снимаем её
        # словами с самой картинки и дописываем в инструкцию. Имеет смысл только при
        # двух референсах: при одном референс И ЕСТЬ субъект, копировать неоткуда.
        self.last_pose_description = None
        if auto_pose and len(images) > 1:
            if processor is None:
                raise ValueError("auto_pose=True требует processor (AutoProcessor Qwen3-VL)")
            self.last_pose_description = describe_pose(self, images[0], processor)
            prompt = append_pose(prompt, self.last_pose_description)

        # Дефолты зависят от чекпоинта: Turbo (TDM) дистиллирован под few-step БЕЗ CFG,
        # Raw — под 28 шагов с guidance 4.5. Жёсткие значения здесь означали бы, что
        # вызов без параметров на Turbo делает ~7x лишней работы (3.5x шагов x 2 за CFG)
        # и портит результат: дистилляция не рассчитана на classifier-free guidance.
        distilled = bool(self.config.is_distilled)
        if num_inference_steps is None:
            num_inference_steps = 8 if distilled else 28
        if guidance_scale is None:
            guidance_scale = 0.0 if distilled else 4.5
        if distilled and guidance_scale > 0:
            warnings.warn(
                f"guidance_scale={guidance_scale} на дистиллированном чекпоинте: "
                "шаг станет вдвое дороже, а качество скорее упадёт — Turbo обучен под 0.0",
                stacklevel=2,
            )

        device = self._execution_device
        multiple = self.vae_scale_factor * self.patch_size          # 16
        height = round_to_multiple(height, multiple)
        width = round_to_multiple(width, multiple)

        self._guidance_scale = guidance_scale
        self._attention_kwargs = None
        self._interrupt = False
        do_cfg = guidance_scale > 0

        # --- 1. условие ---
        if grounding:
            if processor is None:
                raise ValueError("grounding=True требует processor (AutoProcessor Qwen3-VL)")
            prompt_embeds, prompt_mask = encode_grounded(
                self, prompt, images, processor, grounding_px, system_prompt)
            neg_embeds = neg_mask = None
            if do_cfg:
                # В обучении безусловный проход тоже видел картинку — грундим и негатив.
                neg_embeds, neg_mask = encode_grounded(
                    self, negative_prompt or "", images, processor, grounding_px, system_prompt)
        else:
            prompt_embeds, prompt_mask = self.get_text_hidden_states(
                prompt, max_sequence_length, device)
            neg_embeds = neg_mask = None
            if do_cfg:
                neg_embeds, neg_mask = self.get_text_hidden_states(
                    negative_prompt or "", max_sequence_length, device)

        prompt_embeds = prompt_embeds.to(self.transformer.dtype)
        if neg_embeds is not None:
            neg_embeds = neg_embeds.to(self.transformer.dtype)

        # --- 2. референсы (кодируются один раз на весь цикл) ---
        refs_packed, ref_specs = [], []
        for img in images:
            packed, spec = self.encode_reference(img, height, width, fit_mode,
                                                 dtype=self.vae.dtype)
            refs_packed.append(packed.to(self.transformer.dtype))
            ref_specs.append(spec)

        if num_images_per_prompt > 1:
            refs_packed = [r.expand(num_images_per_prompt, -1, -1) for r in refs_packed]
            prompt_embeds = prompt_embeds.expand(num_images_per_prompt, -1, -1, -1)
            prompt_mask = prompt_mask.expand(num_images_per_prompt, -1)
            if neg_embeds is not None:
                neg_embeds = neg_embeds.expand(num_images_per_prompt, -1, -1, -1)
                neg_mask = neg_mask.expand(num_images_per_prompt, -1)

        # --- 3. цель и позиции ---
        num_channels_latents = self.transformer.config.in_channels // (self.patch_size ** 2)
        latents = self.prepare_latents(num_images_per_prompt, num_channels_latents,
                                       height, width, prompt_embeds.dtype, device, generator, None)
        gh = height // multiple
        gw = width // multiple
        position_ids = self.prepare_edit_position_ids(
            prompt_embeds.shape[1], ref_specs, gh, gw, device)

        # --- 4. расписание ---
        # mu считается по длине ЦЕЛИ, а не всей последовательности: расписание описывает
        # зашумление целевого изображения, референсы в нём не участвуют.
        import numpy as np
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        mu = 1.15 if self.config.is_distilled else calculate_shift(
            latents.shape[1],
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 6400),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu)
        self.scheduler.set_begin_index(0)

        # --- 5. опциональный ref_boost ---
        ref_lens = [r.shape[1] for r in refs_packed]
        boosts = [ref_boost_scene] * (len(refs_packed) - 1) + [ref_boost]
        bias = build_ref_bias(
            prompt_embeds.shape[1], ref_lens, latents.shape[1], boosts,
            text_mask=prompt_mask, masks=[None] * (len(refs_packed) - 1) + [boost_mask],
            ref_grids=[(s[0], s[1]) for s in ref_specs],
            device=device, dtype=self.transformer.dtype,
        )
        holder = _BiasHolder(bias)

        # --- 6. денойзинг ---
        def run_loop():
            nonlocal latents
            with self.progress_bar(total=num_inference_steps) as bar:
                for t in timesteps:
                    ts = (t / self.scheduler.config.num_train_timesteps).expand(
                        latents.shape[0]).to(latents.dtype)
                    v = self.edit_velocity(latents, refs_packed, prompt_embeds,
                                           prompt_mask, ts, position_ids)
                    if do_cfg:
                        v_neg = self.edit_velocity(latents, refs_packed, neg_embeds,
                                                   neg_mask, ts, position_ids)
                        # Формула Krea 2: якорь на cond, а не на uncond (ноутбук 01).
                        v = v + guidance_scale * (v - v_neg)
                    latents = self.scheduler.step(v, t, latents, return_dict=False)[0]
                    bar.update()

        # --- 6.1 запуск под edit-LoRA ---
        # Базовые веса Krea 2 — text2image: они копируют референс и не исполняют
        # инструкцию. Навык edit приносит LoRA (см. lora.py); без неё режим
        # технически работает, но редактированием не является.
        if not has_edit_lora(self):
            warnings.warn(
                "edit-LoRA не загружена: базовый Krea 2 скопирует референс и "
                "проигнорирует инструкцию. Загрузи её через "
                "krea2_studio.load_edit_lora(pipe).",
                stacklevel=2,
            )

        with ExitStack() as stack:
            if bias is not None:
                stack.enter_context(ref_boost_active(self.transformer, holder))
            stack.enter_context(edit_lora_active(self, scale=lora_scale))
            run_loop()

        # --- 7. декод ---
        if output_type == "latent":
            return latents

        lat = self._unpack_latents(latents, height, width).to(self.vae.dtype)
        mean = torch.tensor(self.vae.config.latents_mean, device=lat.device, dtype=lat.dtype)
        std = torch.tensor(self.vae.config.latents_std, device=lat.device, dtype=lat.dtype)
        lat = lat * std.view(1, -1, 1, 1, 1) + mean.view(1, -1, 1, 1, 1)
        image = self.vae.decode(lat, return_dict=False)[0][:, :, 0]
        image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()
        return image
