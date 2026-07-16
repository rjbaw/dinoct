# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with the terms
# of the DINOv3 License Agreement, a copy of which is provided in
# LICENSE-DINOV3.md in the root directory of this source tree.

import logging

import numpy as np
import torch
from torch import nn
import torchvision.transforms.v2 as tv

from .transforms import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    Ensure3CH,
    GaussianBlur,
    GaussianNoise,
    PerImageZScore,
)

logger = logging.getLogger("dinoct")


class DataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
        gram_teacher_crops_size=None,
        gram_teacher_no_distortions=False,
        teacher_no_color_jitter=False,
        local_crops_subset_of_global_crops=False,
        patch_size=16,
        share_color_jitter=False,
        horizontal_flips=True,
        solarize_p=0.2,
        solarize_threshold=128,
        gaussian_noise_std=0.0,
        gaussian_noise_p=0.0,
        gaussian_noise_student_only=True,
        aggressive_aug=False,
        aggressive_blur=False,
        aggressive_solarize=False,
        aggressive_jitter=False,
        aggressive_elastic=False,
        aggressive_erasing=False,
        aggressive_noise=False,
        erasing_p=0.0,
        erasing_scale_max=0.2,
        elastic_alpha=0.0,
        elastic_sigma=5.0,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        self.gram_teacher_crops_size = gram_teacher_crops_size
        self.gram_teacher_no_distortions = gram_teacher_no_distortions
        self.teacher_no_color_jitter = teacher_no_color_jitter
        self.local_crops_subset_of_global_crops = local_crops_subset_of_global_crops
        self.patch_size = patch_size
        self.share_color_jitter = share_color_jitter
        self.mean = mean
        self.std = std
        self.solarize_p = float(solarize_p)
        self.solarize_threshold = float(solarize_threshold)
        self.gaussian_noise_std = float(gaussian_noise_std)
        self.gaussian_noise_p = float(gaussian_noise_p)
        self.gaussian_noise_student_only = bool(gaussian_noise_student_only)
        self.aggressive_aug = bool(aggressive_aug)
        self.aggressive_blur = bool(aggressive_blur)
        self.aggressive_solarize = bool(aggressive_solarize)
        self.aggressive_jitter = bool(aggressive_jitter)
        self.aggressive_elastic = bool(aggressive_elastic)
        self.aggressive_erasing = bool(aggressive_erasing)
        self.aggressive_noise = bool(aggressive_noise)
        # MODERATE per-component knobs (override the aggressive_* hardcoded strengths): >0 enables at the
        # given gentle level so all aug TYPES can be on without the aggressive_aug strength that stalled learning.
        self.erasing_p = float(erasing_p)
        self.erasing_scale_max = float(erasing_scale_max)
        self.elastic_alpha = float(elastic_alpha)
        self.elastic_sigma = float(elastic_sigma)

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info(f"gram_crops_size: {gram_teacher_crops_size}")
        logger.info(f"gram_teacher_no_distortions: {gram_teacher_no_distortions}")
        logger.info(f"teacher_no_color_jitter: {teacher_no_color_jitter}")
        logger.info(f"local_crops_subset_of_global_crops: {local_crops_subset_of_global_crops}")
        logger.info(f"patch_size if local_crops_subset_of_global_crops: {patch_size}")
        logger.info(f"share_color_jitter: {share_color_jitter}")
        logger.info(f"horizontal flips: {horizontal_flips}")
        logger.info(f"solarize_p: {self.solarize_p}")
        logger.info(f"solarize_threshold: {self.solarize_threshold}")
        logger.info(f"gaussian_noise_std: {self.gaussian_noise_std}")
        logger.info(f"gaussian_noise_p: {self.gaussian_noise_p}")
        logger.info(f"gaussian_noise_student_only: {self.gaussian_noise_student_only}")

        logger.info("###################################")

        global_crops_max_size = max(global_crops_size, gram_teacher_crops_size if gram_teacher_crops_size else 0)

        self.geometric_augmentation_global = tv.Compose(
            [
                tv.RandomResizedCrop(
                    global_crops_max_size,
                    scale=global_crops_scale,
                    interpolation=tv.InterpolationMode.BICUBIC,
                ),
                tv.RandomHorizontalFlip(p=0.5 if horizontal_flips else 0.0),
            ]
        )

        resize_global = nn.Identity()  # Resize transform applied to global crops before other transforms
        self.resize_global_pre_transf = resize_global
        self.resize_global_post_transf = (
            nn.Identity()
        )  # Resize transform applied to global crops after all other transforms
        self.resize_gram_teacher = None  # Resize transform applied to crops for gram teacher
        if gram_teacher_crops_size is not None:
            self.resize_gram_teacher = tv.Resize(
                gram_teacher_crops_size,
                interpolation=tv.InterpolationMode.BICUBIC,
            )
            # All resize transforms will do nothing if the crop size is already the desired size.
            if gram_teacher_no_distortions:
                resize_global = tv.Resize(
                    global_crops_size,
                    interpolation=tv.InterpolationMode.BICUBIC,
                )
            else:
                self.resize_global_post_transf = tv.Resize(
                    global_crops_size,
                    interpolation=tv.InterpolationMode.BICUBIC,
                )

        self.resize_global_pre_transf = resize_global

        self.geometric_augmentation_local = tv.Compose(
            [
                tv.RandomResizedCrop(
                    local_crops_size,
                    scale=local_crops_scale,
                    interpolation=tv.InterpolationMode.BICUBIC,
                ),
                tv.RandomHorizontalFlip(p=0.5 if horizontal_flips else 0.0),
            ]
        )

        # color distorsions / blurring.
        # aggressive_aug = a stronger augmentation distribution: stronger color-jitter + wider blur
        # + more global-2 blur + more solarize, each applied ONCE in-domain (uint8, before normalize).
        # Default off reproduces the prior behavior exactly.
        # Per-component flags allow decomposing the effect; aggressive_aug = all three.
        agg_jitter = self.aggressive_aug or self.aggressive_jitter
        agg_blur = self.aggressive_aug or self.aggressive_blur
        agg_solarize = self.aggressive_aug or self.aggressive_solarize
        if agg_jitter:
            cj = tv.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.4, hue=0.2)
            cj_p = 0.9
        else:
            cj = tv.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
            cj_p = 0.8
        blur_rmax = 3.0 if agg_blur else 2.0
        g2_blur_p = 0.5 if agg_blur else 0.1
        sol_p = max(self.solarize_p, 0.35) if agg_solarize else self.solarize_p
        color_jittering = tv.Compose(
            [
                tv.RandomApply([cj], p=cj_p),
                tv.RandomGrayscale(p=0.2),
            ]
        )

        global_transfo1_extra = GaussianBlur(p=1.0, radius_max=blur_rmax)

        global_transfo2_extra = tv.Compose(
            [
                GaussianBlur(p=g2_blur_p, radius_max=blur_rmax),
                tv.RandomSolarize(threshold=self.solarize_threshold, p=max(0.0, min(1.0, sol_p))),
            ]
        )

        local_transfo_extra = GaussianBlur(p=0.5, radius_max=blur_rmax)

        normalize_clean_steps = [
            tv.ToImage(),
            tv.ToDtype(torch.float32, scale=True),
        ]
        normalize_noisy_steps = list(normalize_clean_steps)
        _noise_std = self.gaussian_noise_std if self.gaussian_noise_std > 0.0 else (0.1 if self.aggressive_noise else 0.0)
        _noise_p = self.gaussian_noise_p if self.gaussian_noise_p > 0.0 else (0.5 if self.aggressive_noise else 0.0)
        if _noise_std > 0.0 and _noise_p > 0.0:
            normalize_noisy_steps.append(GaussianNoise(std=_noise_std, p=_noise_p))
        _eras_p = self.erasing_p if self.erasing_p > 0.0 else (0.5 if self.aggressive_erasing else 0.0)
        if _eras_p > 0.0:
            # OCT signal-dropout simulation (student-only); on the 1-channel float tensor pre-z-score.
            normalize_noisy_steps.append(tv.RandomErasing(p=_eras_p, scale=(0.02, self.erasing_scale_max), ratio=(0.3, 3.3), value=0.0))
        normalize_tail = [
            Ensure3CH(),
            PerImageZScore(eps=1e-6),
        ]
        # elastic = geometric warp -> synthesizes curvature variety (attacks the high-curvature coverage
        # gap); applied to GLOBAL crops pre-normalize (PIL), survives per-image z-score.
        _el_alpha = self.elastic_alpha if self.elastic_alpha > 0.0 else (50.0 if self.aggressive_elastic else 0.0)
        elastic_pre = [tv.ElasticTransform(alpha=_el_alpha, sigma=self.elastic_sigma)] if _el_alpha > 0.0 else []

        # normalization
        self.normalize = tv.Compose(normalize_noisy_steps + normalize_tail)
        self.normalize_clean = tv.Compose(normalize_clean_steps + normalize_tail)
        if self.share_color_jitter:
            self.color_jittering = color_jittering
            self.global_transfo1_pre = tv.Compose(elastic_pre + [resize_global, global_transfo1_extra])
            self.global_transfo2_pre = tv.Compose(elastic_pre + [resize_global, global_transfo2_extra])
            self.local_transfo_pre = tv.Compose([local_transfo_extra])
        else:
            self.global_transfo1_pre = tv.Compose(elastic_pre + [resize_global, color_jittering, global_transfo1_extra])
            self.global_transfo2_pre = tv.Compose(elastic_pre + [resize_global, color_jittering, global_transfo2_extra])
            self.local_transfo_pre = tv.Compose([color_jittering, local_transfo_extra])

        self.global_transfo1 = tv.Compose([self.global_transfo1_pre, self.normalize])
        self.global_transfo2 = tv.Compose([self.global_transfo2_pre, self.normalize])
        self.local_transfo = tv.Compose([self.local_transfo_pre, self.normalize])
        self.global_transfo1_clean = tv.Compose([self.global_transfo1_pre, self.normalize_clean])
        self.global_transfo2_clean = tv.Compose([self.global_transfo2_pre, self.normalize_clean])
        self.local_transfo_clean = tv.Compose([self.local_transfo_pre, self.normalize_clean])
        self.student_has_private_noise = self.gaussian_noise_student_only and (
            (self.gaussian_noise_std > 0.0 and self.gaussian_noise_p > 0.0)
            or self.aggressive_noise
            or self.aggressive_erasing
            or self.erasing_p > 0.0  # moderate-knob erasing must also stay student-only (teacher targets clean)
        )

    def __call__(self, image):
        output = {}
        output["weak_flag"] = True
        if self.share_color_jitter:
            image = self.color_jittering(image)

        # global crops:
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1_pre = self.global_transfo1_pre(im1_base)
        global_crop_1_transf = self.normalize(global_crop_1_pre)
        global_crop_1_clean = self.normalize_clean(global_crop_1_pre)
        global_crop_1 = self.resize_global_post_transf(global_crop_1_transf)
        global_crop_1_teacher = self.resize_global_post_transf(global_crop_1_clean)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2_pre = self.global_transfo2_pre(im2_base)
        global_crop_2_transf = self.normalize(global_crop_2_pre)
        global_crop_2_clean = self.normalize_clean(global_crop_2_pre)
        global_crop_2 = self.resize_global_post_transf(global_crop_2_transf)
        global_crop_2_teacher = self.resize_global_post_transf(global_crop_2_clean)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        if self.teacher_no_color_jitter:
            output["global_crops_teacher"] = [
                self.resize_global_post_transf(self.normalize_clean(self.resize_global_pre_transf(im1_base))),
                self.resize_global_post_transf(self.normalize_clean(self.resize_global_pre_transf(im2_base))),
            ]
        elif self.student_has_private_noise:
            output["global_crops_teacher"] = [global_crop_1_teacher, global_crop_2_teacher]
        else:
            output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        if self.gram_teacher_crops_size is not None:
            # crops for gram teacher:
            if self.resize_gram_teacher is None:
                raise RuntimeError("Gram teacher crops enabled but resize_gram_teacher not initialized.")

            if self.gram_teacher_no_distortions:
                gram_crop_1 = self.normalize_clean(self.resize_gram_teacher(im1_base))
                gram_crop_2 = self.normalize_clean(self.resize_gram_teacher(im2_base))
            else:
                gram_crop_1_source = global_crop_1_clean if self.student_has_private_noise else global_crop_1_transf
                gram_crop_2_source = global_crop_2_clean if self.student_has_private_noise else global_crop_2_transf
                gram_crop_1 = self.resize_gram_teacher(gram_crop_1_source)
                gram_crop_2 = self.resize_gram_teacher(gram_crop_2_source)
            output["gram_teacher_crops"] = [gram_crop_1, gram_crop_2]

        # local crops:
        if self.local_crops_subset_of_global_crops:
            _local_crops = [self.local_transfo(im1_base) for _ in range(self.local_crops_number // 2)] + [
                self.local_transfo(im2_base) for _ in range(self.local_crops_number // 2)
            ]

            local_crops = []
            offsets = []
            gs = self.global_crops_size
            ls = self.local_crops_size
            for img in _local_crops:
                rx, ry = np.random.randint(0, (gs - ls) // self.patch_size, 2) * self.patch_size
                local_crops.append(img[:, rx : rx + ls, ry : ry + ls])
                offsets.append((rx, ry))

            output["local_crops"] = local_crops
            output["offsets"] = offsets
        else:
            local_crops = [
                self.local_transfo(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
            ]
            output["local_crops"] = local_crops
            output["offsets"] = ()

        return output
