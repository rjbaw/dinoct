import logging

import numpy as np
import torch
from torch import nn
import torchvision.transforms.v2 as tv

from .transforms import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    GaussianBlur,
    make_normalize_transform,
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

        resize_global = nn.Identity()  # Resize transform applied to global crops after random crop
        self.resize_global_post_transf = (
            nn.Identity()
        )  # Resize transform applied to global crops after all other transforms
        self.resize_gram_teacher = None  # Resize transform applied to crops for gram teacher
        if gram_teacher_crops_size is not None:
            # All resize transforms will do nothing if the crop size is already the desired size.
            if gram_teacher_no_distortions:
                # When there a no distortions for the gram teacher crop, we can resize before the distortions.
                # This is the preferred order, because it keeps the image size for the augmentations consistent,
                # which matters e.g. for GaussianBlur.
                resize_global = tv.Resize(
                    global_crops_size,
                    interpolation=tv.InterpolationMode.BICUBIC,
                )
            else:
                # When there a no distortions for the gram teacher crop, we need to resize after the distortions,
                # because the distortions are shared between global and gram teacher crops.
                self.resize_global_post_transf = tv.Resize(
                    global_crops_size,
                    interpolation=tv.InterpolationMode.BICUBIC,
                )

            self.resize_gram_teacher = tv.Resize(
                gram_teacher_crops_size,
                interpolation=tv.InterpolationMode.BICUBIC,
            )

        self.geometric_augmentation_local = tv.Compose(
            [
                tv.RandomResizedCrop(
                    local_crops_size,
                    scale=local_crops_scale,
                    interpolation=tv.InterpolationMode.BICUBIC,
                ),
                tv.RandomHorizontalFlip(p=0.5),
            ]
        )

        # color distorsions / blurring
        color_jittering = tv.Compose(
            [
                tv.RandomApply(
                    [tv.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                tv.RandomGrayscale(p=0.2),
            ]
        )

        global_transfo1_extra = GaussianBlur(p=1.0)

        global_transfo2_extra = tv.Compose(
            [
                GaussianBlur(p=0.1),
                tv.RandomSolarize(threshold=0.5, p=0.2),
            ]
        )

        local_transfo_extra = GaussianBlur(p=0.5)

        # normalization
        self.normalize = tv.Compose(
            [
                tv.ToImage(),
                tv.ToDtype(torch.float32, scale=True),
                make_normalize_transform(mean=mean, std=std),
            ]
        )
        if self.share_color_jitter:
            self.color_jittering = color_jittering
            self.global_transfo1 = tv.Compose([resize_global, global_transfo1_extra, self.normalize])
            self.global_transfo2 = tv.Compose([resize_global, global_transfo2_extra, self.normalize])
            self.local_transfo = tv.Compose([local_transfo_extra, self.normalize])
        else:
            self.global_transfo1 = tv.Compose([resize_global, color_jittering, global_transfo1_extra, self.normalize])
            self.global_transfo2 = tv.Compose([resize_global, color_jittering, global_transfo2_extra, self.normalize])
            self.local_transfo = tv.Compose([color_jittering, local_transfo_extra, self.normalize])

    def __call__(self, image):
        output = {}
        output["weak_flag"] = True
        if self.share_color_jitter:
            image = self.color_jittering(image)

        # global crops:
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1_transf = self.global_transfo1(im1_base)
        global_crop_1 = self.global_transfo1(global_crop_1_transf)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2_transf = self.global_transfo2(im2_base)
        global_crop_2 = self.global_transfo2(global_crop_2_transf)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        if self.teacher_no_color_jitter:
            output["global_crops_teacher"] = [
                self.normalize(im1_base),
                self.normalize(im2_base),
            ]
        else:
            output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        if self.gram_teacher_crops_size is not None:
            # crops for gram teacher:
            if self.resize_gram_teacher is None:
                raise RuntimeError("Gram teacher crops enabled but resize_gram_teacher not initialized.")

            if self.gram_teacher_no_distortions:
                gram_crop_1 = self.normalize(self.resize_gram_teacher(im1_base))
                gram_crop_2 = self.normalize(self.resize_gram_teacher(im2_base))
            else:
                gram_crop_1 = self.resize_gram_teacher(global_crop_1_transf)
                gram_crop_2 = self.resize_gram_teacher(global_crop_2_transf)
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
