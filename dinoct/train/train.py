from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import amp, nn, optim
import yaml

from ..data import DataAugmentationDINO, MaskingGenerator, collate_data_and_cast, make_data_loader, make_dataset
from ..layers import DINOHead
from ..loss import DINOLoss, KoLeoLoss, iBOTPatchLoss
from ..models import build_backbone
from ..utils import fix_random_seeds
from .core.schedules import cosine_schedule, linear_warmup_cosine_decay
from .param_groups import fuse_params_groups, get_params_groups_with_decay_fsdp
from .post_train import run_post_training

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "ssl_default_config.yaml"
DEFAULT_TRAIN_CONFIG = REPO_ROOT / "configs" / "train" / "oct.yaml"

logger = logging.getLogger("dinoct")


def deep_update(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = deep_update(base[key], value)
        else:
            base[key] = value
    return base


def get_cfg(cfg: dict[str, Any], path: tuple[str, ...], default: Any) -> Any:
    cur: Any = cfg
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def load_training_cfg(config_path: Path | None) -> dict[str, Any]:
    if not DEFAULT_CONFIG.exists():
        raise FileNotFoundError(DEFAULT_CONFIG)
    base = yaml.safe_load(DEFAULT_CONFIG.read_text())
    if not isinstance(base, dict):
        raise ValueError(f"Invalid base config at {DEFAULT_CONFIG}")
    if config_path:
        override = yaml.safe_load(config_path.read_text())
        if isinstance(override, dict):
            base = deep_update(base, override)
    return base


def resolve_dataset_path(dataset_str: str) -> str:
    """
    Best-effort resolution so relative OCT paths work when launched from repo root.
    """
    parts = dataset_str.split(":")
    tokens: dict[str, str] = {}
    for token in parts[1:]:
        if "=" in token:
            k, v = token.split("=", 1)
            tokens[k] = v
    for key in ("root", "extra"):
        if key in tokens:
            path = Path(tokens[key])
            if not path.exists():
                candidates = [
                    REPO_ROOT / tokens[key],
                    REPO_ROOT / "data" / tokens[key],
                    REPO_ROOT.parent / tokens[key],
                ]
                for cand in candidates:
                    if cand.exists():
                        tokens[key] = str(cand.resolve())
                        break
    rebuilt = [parts[0]] + [f"{k}={v}" for k, v in tokens.items()]
    return ":".join(rebuilt) if tokens else dataset_str


def build_dataloader(cfg: dict[str, Any]) -> tuple[torch.utils.data.DataLoader, int]:
    global_size = int(get_cfg(cfg, ("crops", "global_crops_size"), 224))
    local_size = int(get_cfg(cfg, ("crops", "local_crops_size"), 96))
    local_num = int(get_cfg(cfg, ("crops", "local_crops_number"), 8))
    patch_size = int(get_cfg(cfg, ("student", "patch_size"), 14))
    dataset_str = resolve_dataset_path(
        str(get_cfg(cfg, ("train", "dataset_path"), "OCT:root=data/oct:extra=data/oct/extra"))
    )

    augment = DataAugmentationDINO(
        get_cfg(cfg, ("crops", "global_crops_scale"), (0.32, 1.0)),
        get_cfg(cfg, ("crops", "local_crops_scale"), (0.05, 0.32)),
        local_num,
        global_crops_size=global_size,
        local_crops_size=local_size,
        patch_size=patch_size,
        share_color_jitter=bool(get_cfg(cfg, ("train", "share_color_jitter"), False)),
        horizontal_flips=bool(get_cfg(cfg, ("train", "horizontal_flips"), True)),
    )

    mask_gen = MaskingGenerator(
        input_size=(global_size // patch_size, global_size // patch_size),
        max_num_patches=int(0.5 * (global_size // patch_size) ** 2),
    )
    n_tokens = (global_size // patch_size) ** 2
    collate_fn = lambda batch: collate_data_and_cast(  # noqa: E731
        batch,
        mask_ratio_tuple=tuple(get_cfg(cfg, ("ibot", "mask_ratio_min_max"), (0.1, 0.5))),
        mask_probability=float(get_cfg(cfg, ("ibot", "mask_sample_probability"), 0.5)),
        n_tokens=n_tokens,
        mask_generator=mask_gen,
        dtype=torch.float32,
    )

    dataset = make_dataset(dataset_str=dataset_str, transform=augment, target_transform=lambda _: ())
    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=int(get_cfg(cfg, ("train", "batch_size_per_gpu"), 8)),
        num_workers=int(get_cfg(cfg, ("train", "num_workers"), 4)),
        shuffle=True,
        seed=int(get_cfg(cfg, ("train", "seed"), 0)),
        drop_last=True,
        collate_fn=collate_fn,
    )
    return data_loader, n_tokens


def _resolve_ffn_layer(name: str) -> str:
    if name.lower() in {"swiglufused", "swiglu_fused"}:
        return "swiglu"
    return name


@dataclass
class ModelBundle:
    student: nn.Module
    teacher: nn.Module
    student_head: DINOHead
    teacher_head: DINOHead
    student_ibot_head: DINOHead | None
    teacher_ibot_head: DINOHead | None


def build_models(cfg: dict[str, Any], device: torch.device) -> ModelBundle:
    arch_raw = str(get_cfg(cfg, ("student", "arch"), "vit_small"))
    if arch_raw.startswith("vit_"):
        arch = arch_raw.replace("vit_", "")
    else:
        arch = arch_raw
    patch_size = int(get_cfg(cfg, ("student", "patch_size"), 14))
    drop_path_rate = float(get_cfg(cfg, ("student", "drop_path_rate"), 0.0))
    layerscale = get_cfg(cfg, ("student", "layerscale"), None)
    ffn_layer = _resolve_ffn_layer(str(get_cfg(cfg, ("student", "ffn_layer"), "mlp")))

    student = build_backbone(
        arch,
        patch_size=patch_size,
        drop_path_rate=drop_path_rate,
        layerscale_init=layerscale,
        ffn_layer=ffn_layer,
        qkv_bias=bool(get_cfg(cfg, ("student", "qkv_bias"), True)),
        proj_bias=bool(get_cfg(cfg, ("student", "proj_bias"), True)),
        ffn_bias=bool(get_cfg(cfg, ("student", "ffn_bias"), True)),
        n_storage_tokens=int(get_cfg(cfg, ("student", "num_register_tokens"), 0)),
        mask_k_bias=bool(get_cfg(cfg, ("student", "mask_k_bias"), False)),
        device=device,
    ).to(device)
    teacher = build_backbone(
        arch,
        patch_size=patch_size,
        drop_path_rate=drop_path_rate,
        layerscale_init=layerscale,
        ffn_layer=ffn_layer,
        qkv_bias=bool(get_cfg(cfg, ("student", "qkv_bias"), True)),
        proj_bias=bool(get_cfg(cfg, ("student", "proj_bias"), True)),
        ffn_bias=bool(get_cfg(cfg, ("student", "ffn_bias"), True)),
        n_storage_tokens=int(get_cfg(cfg, ("student", "num_register_tokens"), 0)),
        mask_k_bias=bool(get_cfg(cfg, ("student", "mask_k_bias"), False)),
        device=device,
    ).to(device)
    teacher.load_state_dict(student.state_dict(), strict=False)
    for p in teacher.parameters():
        p.requires_grad = False

    dino_out = int(get_cfg(cfg, ("dino", "head_n_prototypes"), 65536))
    dino_hidden = int(get_cfg(cfg, ("dino", "head_hidden_dim"), 2048))
    dino_bottleneck = int(get_cfg(cfg, ("dino", "head_bottleneck_dim"), 256))
    dino_nlayers = int(get_cfg(cfg, ("dino", "head_nlayers"), 3))

    student_head = DINOHead(
        in_dim=student.embed_dim,
        out_dim=dino_out,
        hidden_dim=dino_hidden,
        bottleneck_dim=dino_bottleneck,
        nlayers=dino_nlayers,
    ).to(device)
    teacher_head = DINOHead(
        in_dim=teacher.embed_dim,
        out_dim=dino_out,
        hidden_dim=dino_hidden,
        bottleneck_dim=dino_bottleneck,
        nlayers=dino_nlayers,
    ).to(device)
    teacher_head.load_state_dict(student_head.state_dict(), strict=False)
    for p in teacher_head.parameters():
        p.requires_grad = False

    ibot_separate = bool(get_cfg(cfg, ("ibot", "separate_head"), False))
    ibot_out = int(get_cfg(cfg, ("ibot", "head_n_prototypes"), dino_out))
    ibot_hidden = int(get_cfg(cfg, ("ibot", "head_hidden_dim"), dino_hidden))
    ibot_bottleneck = int(get_cfg(cfg, ("ibot", "head_bottleneck_dim"), dino_bottleneck))
    ibot_nlayers = int(get_cfg(cfg, ("ibot", "head_nlayers"), dino_nlayers))
    student_ibot_head = None
    teacher_ibot_head = None
    if ibot_separate:
        student_ibot_head = DINOHead(
            in_dim=student.embed_dim,
            out_dim=ibot_out,
            hidden_dim=ibot_hidden,
            bottleneck_dim=ibot_bottleneck,
            nlayers=ibot_nlayers,
        ).to(device)
        teacher_ibot_head = DINOHead(
            in_dim=teacher.embed_dim,
            out_dim=ibot_out,
            hidden_dim=ibot_hidden,
            bottleneck_dim=ibot_bottleneck,
            nlayers=ibot_nlayers,
        ).to(device)
        teacher_ibot_head.load_state_dict(student_ibot_head.state_dict(), strict=False)
        for p in teacher_ibot_head.parameters():
            p.requires_grad = False

    return ModelBundle(
        student=student,
        teacher=teacher,
        student_head=student_head,
        teacher_head=teacher_head,
        student_ibot_head=student_ibot_head,
        teacher_ibot_head=teacher_ibot_head,
    )


@dataclass
class Schedules:
    lr: list[float]
    weight_decay: list[float]
    teacher_temp: list[float]
    momentum: list[float]
    freeze_last_layer_iters: int


def build_optimizer_and_schedules(
    cfg: dict[str, Any],
    bundle: ModelBundle,
    total_iters: int,
) -> tuple[optim.Optimizer, Schedules]:
    batch_size = int(get_cfg(cfg, ("train", "batch_size_per_gpu"), 8))
    scaling_rule = str(get_cfg(cfg, ("optim", "scaling_rule"), "sqrt_wrt_1024")).lower()
    if scaling_rule.startswith("sqrt"):
        scaled_lr = float(get_cfg(cfg, ("optim", "base_lr"), 1.5e-4)) * (batch_size / 1024) ** 0.5
    else:
        scaled_lr = float(get_cfg(cfg, ("optim", "base_lr"), 1.5e-4)) * (batch_size / 256)

    epoch_length = int(get_cfg(cfg, ("train", "OFFICIAL_EPOCH_LENGTH"), 1250))
    warmup_iters = int(get_cfg(cfg, ("optim", "warmup_epochs"), 10) * epoch_length)
    min_lr = float(get_cfg(cfg, ("optim", "min_lr"), 1e-6))
    lr_schedule = linear_warmup_cosine_decay(
        start=0.0,
        peak=scaled_lr,
        end=min_lr,
        warmup_iterations=warmup_iters,
        total_iterations=total_iters,
    )

    wd = float(get_cfg(cfg, ("optim", "weight_decay"), 0.04))
    wd_end = float(get_cfg(cfg, ("optim", "weight_decay_end"), wd))
    wd_schedule = cosine_schedule(wd, wd_end, total_iters)

    teacher_m = float(get_cfg(cfg, ("teacher", "momentum_teacher"), 0.996))
    teacher_final_m = float(get_cfg(cfg, ("teacher", "final_momentum_teacher"), 1.0))
    momentum_schedule = cosine_schedule(teacher_m, teacher_final_m, total_iters)

    teacher_temp = float(get_cfg(cfg, ("teacher", "teacher_temp"), 0.07))
    warm_temp = float(get_cfg(cfg, ("teacher", "warmup_teacher_temp"), 0.04))
    warm_temp_epochs = int(get_cfg(cfg, ("teacher", "warmup_teacher_temp_epochs"), 0))
    warm_temp_iters = warm_temp_epochs * epoch_length
    teacher_temp_schedule = linear_warmup_cosine_decay(
        start=warm_temp,
        peak=teacher_temp,
        end=teacher_temp,
        warmup_iterations=warm_temp_iters,
        total_iterations=total_iters,
        cosine_iterations=max(total_iters - warm_temp_iters, 1),
    )

    backbone_param_groups = get_params_groups_with_decay_fsdp(
        bundle.student,
        layerwise_decay=float(get_cfg(cfg, ("optim", "layerwise_decay"), 1.0)),
        patch_embed_lr_mult=float(get_cfg(cfg, ("optim", "patch_embed_lr_mult"), 1.0)),
    )
    param_groups = fuse_params_groups(backbone_param_groups)

    head_groups = [
        {"params": bundle.student_head.parameters(), "weight_decay": 1.0, "lr_multiplier": 1.0},
    ]
    if bundle.student_ibot_head is not None:
        head_groups.append({"params": bundle.student_ibot_head.parameters(), "weight_decay": 1.0, "lr_multiplier": 1.0})
    param_groups.extend(head_groups)

    for group in param_groups:
        group.setdefault("lr_multiplier", 1.0)
        group["weight_decay_factor"] = float(group.get("weight_decay", 1.0))
        group["lr"] = 0.0
        group["weight_decay"] = wd * group["weight_decay_factor"]

    optimizer = optim.AdamW(
        param_groups,
        lr=scaled_lr,
        betas=(
            float(get_cfg(cfg, ("optim", "adamw_beta1"), 0.9)),
            float(get_cfg(cfg, ("optim", "adamw_beta2"), 0.999)),
        ),
    )

    freeze_epochs = int(get_cfg(cfg, ("optim", "freeze_last_layer_epochs"), 1))
    schedules = Schedules(
        lr=lr_schedule,
        weight_decay=wd_schedule,
        teacher_temp=teacher_temp_schedule,
        momentum=momentum_schedule,
        freeze_last_layer_iters=freeze_epochs * epoch_length,
    )
    return optimizer, schedules


def update_teacher_weights(student: nn.Module, teacher: nn.Module, momentum: float) -> None:
    with torch.no_grad():
        for ps, pt in zip(student.parameters(), teacher.parameters()):
            pt.data.mul_(momentum).add_(ps.data, alpha=1.0 - momentum)


def apply_freeze_last_layer(head: DINOHead, iteration: int, freeze_until: int) -> None:
    requires_grad = iteration >= freeze_until
    for p in head.last_layer.parameters():
        p.requires_grad = requires_grad


def setup_file_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(formatter)
    fh.setLevel(logging.INFO)
    root_logger = logging.getLogger()
    # Avoid duplicate handlers if re-run in same process
    for h in root_logger.handlers:
        if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == str(log_path):
            break
    else:
        root_logger.addHandler(fh)


def maybe_compile(module: nn.Module, enabled: bool, backend: str | None = None) -> nn.Module:
    if not enabled:
        return module
    if not hasattr(torch, "compile"):
        logger.warning("torch.compile not available; skipping compilation")
        return module
    try:
        # Avoid global import-time side effects; tune dynamo only if compile is requested.
        if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "config"):
            try:
                torch._dynamo.config.automatic_dynamic_shapes = False
                torch._dynamo.config.accumulated_cache_size_limit = 1024
            except Exception:  # pragma: no cover
                pass
        return torch.compile(module, backend=backend)
    except Exception as exc:  # pragma: no cover - best-effort compile
        logger.warning("torch.compile failed (%s); continuing without compilation", exc)
        return module


def train(
    cfg: dict[str, Any],
    *,
    steps_override: int | None = None,
    output_dir_override: Path | None = None,
    seed_override: int | None = None,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DINOv3 training.")
    device = torch.device("cuda")
    fix_random_seeds(int(seed_override if seed_override is not None else get_cfg(cfg, ("train", "seed"), 0)))

    data_loader, _ = build_dataloader(cfg)
    bundle = build_models(cfg, device)
    compile_enabled = bool(get_cfg(cfg, ("train", "compile"), False))
    compile_backend = get_cfg(cfg, ("train", "compile_backend"), None)
    bundle.student = maybe_compile(bundle.student, compile_enabled, compile_backend)
    bundle.student_head = maybe_compile(bundle.student_head, compile_enabled, compile_backend)
    if bundle.student_ibot_head is not None:
        bundle.student_ibot_head = maybe_compile(bundle.student_ibot_head, compile_enabled, compile_backend)

    epoch_length = int(get_cfg(cfg, ("train", "OFFICIAL_EPOCH_LENGTH"), 1250))
    epochs = int(get_cfg(cfg, ("optim", "epochs"), 100))
    total_iters = steps_override if steps_override is not None else epochs * epoch_length

    optimizer, schedules = build_optimizer_and_schedules(cfg, bundle, total_iters)

    dino_loss = DINOLoss(
        out_dim=int(get_cfg(cfg, ("dino", "head_n_prototypes"), 65536)),
        student_temp=float(get_cfg(cfg, ("dino", "student_temp"), 0.1)),
    ).to(device)
    ibot_loss = iBOTPatchLoss(
        patch_out_dim=int(
            get_cfg(cfg, ("ibot", "head_n_prototypes"), get_cfg(cfg, ("dino", "head_n_prototypes"), 65536))
        ),
        student_temp=float(get_cfg(cfg, ("ibot", "student_temp"), 0.1)),
    ).to(device)
    koleo_loss = KoLeoLoss().to(device)

    scaler = amp.GradScaler("cuda", enabled=bool(get_cfg(cfg, ("compute_precision", "grad_scaler"), True)))
    centering = str(get_cfg(cfg, ("train", "centering"), "centering")).lower()
    output_dir = (
        Path(output_dir_override) if output_dir_override else Path(get_cfg(cfg, ("train", "output_dir"), "outputs"))
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    pretrain_dir = output_dir / "pretrain"
    pretrain_dir.mkdir(parents=True, exist_ok=True)
    setup_file_logging(pretrain_dir / "train.log")
    metrics_path = pretrain_dir / "metrics.csv"
    metrics_fh = metrics_path.open("a", newline="")
    metrics_writer = csv.writer(metrics_fh)
    if metrics_path.stat().st_size == 0:
        metrics_writer.writerow(["step", "loss", "dino", "ibot", "lr", "weight_decay", "teacher_temp", "momentum"])

    logger.info("starting DINOv3 training: %d iterations (~%d epochs)", total_iters, total_iters // epoch_length)
    iterator = iter(data_loader)
    for iteration in range(total_iters):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(data_loader)
            batch = next(iterator)

        lr = schedules.lr[iteration]
        weight_decay = schedules.weight_decay[iteration]
        momentum = schedules.momentum[iteration]
        teacher_temp = schedules.teacher_temp[iteration]

        for group in optimizer.param_groups:
            group["lr"] = lr * group.get("lr_multiplier", 1.0)
            group["weight_decay"] = weight_decay * group.get("weight_decay_factor", 1.0)

        apply_freeze_last_layer(bundle.student_head, iteration, schedules.freeze_last_layer_iters)
        if bundle.student_ibot_head is not None:
            apply_freeze_last_layer(bundle.student_ibot_head, iteration, schedules.freeze_last_layer_iters)

        global_crops = batch["collated_global_crops"].to(device, non_blocking=True)
        local_crops = batch["collated_local_crops"].to(device, non_blocking=True)
        masks = batch["collated_masks"].to(device, non_blocking=True)
        mask_indices_list = batch["mask_indices_list"].to(device, non_blocking=True)
        masks_weight = batch["masks_weight"].to(device, non_blocking=True)
        n_masked = int(mask_indices_list.numel())

        global_list = list(global_crops.chunk(2))
        local_list = list(local_crops.chunk(int(get_cfg(cfg, ("crops", "local_crops_number"), 8))))
        global_masks = list(masks.chunk(2))
        mask_list = global_masks + [None for _ in local_list]

        with torch.no_grad():
            teacher_outputs = bundle.teacher.forward_features_list(global_list, [None for _ in global_list])
            teacher_cls = [o["x_norm_clstoken"] for o in teacher_outputs]
            teacher_patches = [o["x_norm_patchtokens"] for o in teacher_outputs]
            teacher_logits = [bundle.teacher_head(t) for t in teacher_cls]
            teacher_concat = torch.cat(teacher_logits, dim=0)
            if centering == "sinkhorn_knopp":
                teacher_targets = dino_loss.sinkhorn_knopp_teacher(teacher_concat, teacher_temp)
            else:
                teacher_targets = dino_loss.softmax_center_teacher(teacher_concat, teacher_temp)
            dino_loss.update_center(teacher_concat)
            teacher_targets_split = torch.split(teacher_targets, teacher_cls[0].shape[0], dim=0)

            if n_masked > 0:
                teacher_patch_tokens = torch.cat([t.flatten(0, 1) for t in teacher_patches], dim=0)
                teacher_masked = teacher_patch_tokens.index_select(dim=0, index=mask_indices_list)
                ibot_teacher_head = bundle.teacher_ibot_head or bundle.teacher_head
                teacher_masked_logits = ibot_teacher_head(teacher_masked)
                if centering == "sinkhorn_knopp":
                    n_masked_tensor = batch.get("n_masked_patches", torch.tensor([n_masked], device=device)).to(device)
                    teacher_ibot_targets = ibot_loss.sinkhorn_knopp_teacher(
                        teacher_masked_logits,
                        teacher_temp,
                        n_masked_patches_tensor=n_masked_tensor,
                    )
                else:
                    teacher_ibot_targets = ibot_loss.softmax_center_teacher(
                        teacher_masked_logits.unsqueeze(0),
                        teacher_temp,
                    ).squeeze(0)
                ibot_loss.update_center(teacher_masked_logits.unsqueeze(0))
            else:
                teacher_ibot_targets = None

        with amp.autocast(device_type="cuda", enabled=scaler.is_enabled()):
            student_outputs = bundle.student.forward_features_list(global_list + local_list, mask_list)
            student_cls = [o["x_norm_clstoken"] for o in student_outputs]
            student_patches = [o["x_norm_patchtokens"] for o in student_outputs[:2]]  # mask applied only to globals

            student_logits = [bundle.student_head(t) for t in student_cls]
            dino_term = dino_loss(student_logits, list(teacher_targets_split)) / len(student_logits)
            loss = float(get_cfg(cfg, ("dino", "loss_weight"), 1.0)) * dino_term

            koleo_weight = float(get_cfg(cfg, ("dino", "koleo_loss_weight"), 0.0))
            if koleo_weight > 0:
                loss = loss + koleo_weight * koleo_loss(teacher_concat)

            ibot_weight = float(get_cfg(cfg, ("ibot", "loss_weight"), 1.0))
            ibot_term = torch.tensor(0.0, device=device)
            if ibot_weight > 0 and n_masked > 0 and teacher_ibot_targets is not None:
                student_patch_tokens = torch.cat([t.flatten(0, 1) for t in student_patches], dim=0)
                student_masked = student_patch_tokens.index_select(dim=0, index=mask_indices_list)
                ibot_student_head = bundle.student_ibot_head or bundle.student_head
                student_masked_logits = ibot_student_head(student_masked)
                ibot_term = ibot_loss.forward_masked(
                    student_masked_logits,
                    teacher_ibot_targets,
                    student_masks_flat=masks,
                    n_masked_patches=n_masked,
                    masks_weight=masks_weight,
                )
                loss = loss + ibot_weight * ibot_term

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        clip_grad = float(get_cfg(cfg, ("optim", "clip_grad"), 0.0))
        if clip_grad > 0:
            scaler.unscale_(optimizer)
            params_to_clip = [p for group in optimizer.param_groups for p in group["params"] if p.requires_grad]
            nn.utils.clip_grad_norm_(params_to_clip, max_norm=clip_grad)
        scaler.step(optimizer)
        scaler.update()

        update_teacher_weights(bundle.student, bundle.teacher, momentum)
        update_teacher_weights(bundle.student_head, bundle.teacher_head, momentum)
        if bundle.student_ibot_head is not None and bundle.teacher_ibot_head is not None:
            update_teacher_weights(bundle.student_ibot_head, bundle.teacher_ibot_head, momentum)

        if (iteration + 1) % max(1, int(get_cfg(cfg, ("train", "log_every"), 10))) == 0 or iteration == 0:
            metrics_writer.writerow(
                [
                    iteration + 1,
                    float(loss.detach()),
                    float(dino_term.detach()),
                    float(ibot_term.detach()),
                    lr,
                    weight_decay,
                    teacher_temp,
                    momentum,
                ]
            )
            metrics_fh.flush()
            logger.info(
                "[%05d/%05d] loss=%.4f dino=%.4f ibot=%.4f lr=%.6f wd=%.4f temp=%.3f mom=%.4f",
                iteration + 1,
                total_iters,
                float(loss.detach()),
                float(dino_term.detach()),
                float(ibot_term.detach()),
                lr,
                weight_decay,
                teacher_temp,
                momentum,
            )

    pretrain_dir = output_dir / "pretrain"
    pretrain_dir.mkdir(parents=True, exist_ok=True)
    pretrain_path = pretrain_dir / "dinov3_pretrain.pth"
    torch.save(
        {
            "student": bundle.student.state_dict(),
            "student_head": bundle.student_head.state_dict(),
            "student_ibot_head": bundle.student_ibot_head.state_dict() if bundle.student_ibot_head else None,
            "teacher": bundle.teacher.state_dict(),
            "teacher_head": bundle.teacher_head.state_dict(),
            "teacher_ibot_head": bundle.teacher_ibot_head.state_dict() if bundle.teacher_ibot_head else None,
            "config": cfg,
        },
        pretrain_path,
    )
    (pretrain_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    metrics_fh.close()
    logger.info("training complete. checkpoint saved to %s", pretrain_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("DINOv3 OCT training (single GPU)")
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_TRAIN_CONFIG, help="YAML config to merge on top of defaults"
    )
    parser.add_argument("--output-dir", type=Path, default="outputs", help="Override output directory")
    parser.add_argument("--steps", type=int, default=None, help="Override total iteration count")
    parser.add_argument("--batch-size", type=int, default=None, help="Override per-GPU batch size")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    # Post-train (curve) stage
    parser.add_argument(
        "--post-train-steps", type=int, default=None, help="Run curve post-training for N steps (0 to skip)"
    )
    parser.add_argument("--post-train-batch-size", type=int, default=None, help="Batch size for post-training")
    parser.add_argument("--post-train-lr-head", type=float, default=None)
    parser.add_argument("--post-train-lr-lora", type=float, default=None)
    parser.add_argument("--post-train-wd-head", type=float, default=None)
    parser.add_argument("--post-train-wd-lora", type=float, default=None)
    parser.add_argument("--post-train-sigma", type=float, default=None)
    parser.add_argument("--post-train-lambda-bg", type=float, default=None)
    parser.add_argument("--post-train-lambda-curve", type=float, default=None)
    parser.add_argument("--post-train-lambda-curv", type=float, default=None)
    parser.add_argument("--post-train-lora-blocks", type=int, default=None)
    parser.add_argument("--post-train-lora-r", type=int, default=None)
    parser.add_argument("--post-train-lora-alpha", type=int, default=None)
    parser.add_argument("--post-train-lora-dropout", type=float, default=None)
    parser.add_argument("--post-train-lora-use-mlp", action="store_true")
    parser.add_argument(
        "--post-train-only",
        action="store_true",
        help="Skip SSL pretrain; load --pretrained-backbone and post-train only",
    )
    parser.add_argument(
        "--pretrained-backbone", type=Path, default=None, help="Backbone checkpoint to load for --post-train-only"
    )
    args = parser.parse_args(argv)
    if args.pretrained_backbone is not None and not args.post_train_only:
        parser.error("--pretrained-backbone is only valid with --post-train-only")
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    cfg = load_training_cfg(args.config)
    post_cfg = cfg.get("post_train", {})
    if args.batch_size is not None:
        cfg.setdefault("train", {})["batch_size_per_gpu"] = args.batch_size
    if args.output_dir is not None:
        cfg.setdefault("train", {})["output_dir"] = str(args.output_dir)
    # Optional SSL pretrain
    output_dir = Path(args.output_dir) if args.output_dir else Path(get_cfg(cfg, ("train", "output_dir"), "outputs"))
    # Post-train curve stage if requested
    post_steps = args.post_train_steps if args.post_train_steps is not None else int(post_cfg.get("steps", 0))
    if args.post_train_only and post_steps <= 0:
        raise SystemExit(
            "No work to do: --post-train-only was set but post-train steps is 0.\n"
            "Pass --post-train-steps N, or set post_train.steps > 0 in your config."
        )
    if args.post_train_only:
        ckpt_path = args.pretrained_backbone or (output_dir / "pretrain" / "dinov3_pretrain.pth")
        if not ckpt_path.exists():
            raise SystemExit(
                f"Pretrained backbone checkpoint not found: {ckpt_path}\n"
                "Pass --pretrained-backbone PATH, or run pretraining first to create outputs/.../pretrain/dinov3_pretrain.pth."
            )

    if not args.post_train_only:
        train(cfg, steps_override=args.steps, output_dir_override=output_dir, seed_override=args.seed)
    if post_steps and post_steps > 0:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for curve post-training.")
        device = torch.device("cuda")
        post_dir = output_dir / "post_train"
        setup_file_logging(post_dir / "post_train.log")
        # Load backbone weights from checkpoint if provided, else from freshly trained student
        backbone = None
        pretrain_path = output_dir / "pretrain" / "dinov3_pretrain.pth"
        ckpt_path = args.pretrained_backbone if (args.post_train_only and args.pretrained_backbone) else pretrain_path
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("student", ckpt.get("model", ckpt))
        arch_raw = str(get_cfg(cfg, ("student", "arch"), "vit_small"))
        arch = arch_raw.replace("vit_", "") if arch_raw.startswith("vit_") else arch_raw
        patch_size = int(get_cfg(cfg, ("student", "patch_size"), 14))
        backbone = build_backbone(arch, patch_size=patch_size)
        backbone.load_state_dict(state, strict=False)
        backbone.to(device)
        for p in backbone.parameters():
            p.requires_grad = False

        resolved_ds = resolve_dataset_path(
            get_cfg(cfg, ("train", "dataset_path"), "OCT:root=data/oct:extra=data/oct/extra")
        )
        post_dir = output_dir / "post_train"
        post_dir.mkdir(parents=True, exist_ok=True)
        post_out = post_dir / "fused_curve.pth"
        best_out = post_dir / "fused_curve_best.pth"
        sam_cfg = post_cfg.get("sam", {})
        if not isinstance(sam_cfg, dict):
            sam_cfg = {}
        sam_rho = sam_cfg.get("rho", post_cfg.get("sam_rho", 0.05))
        run_post_training(
            backbone=backbone,
            patch_size=int(get_cfg(cfg, ("student", "patch_size"), 14)),
            dataset_str=resolved_ds,
            seed=int(get_cfg(cfg, ("train", "seed"), 0)),
            steps=int(post_steps),
            batch_size=int(args.post_train_batch_size or post_cfg.get("batch_size", 128)),
            num_workers=int(get_cfg(cfg, ("train", "num_workers"), 4)),
            lr_head=float(args.post_train_lr_head or post_cfg.get("lr_head", 1e-3)),
            wd_head=float(args.post_train_wd_head or post_cfg.get("wd_head", 5e-4)),
            lr_lora=float(args.post_train_lr_lora or post_cfg.get("lr_lora", 5e-4)),
            wd_lora=float(args.post_train_wd_lora or post_cfg.get("wd_lora", 0.0)),
            sigma=float(args.post_train_sigma or post_cfg.get("sigma", 1.5)),
            lambda_bg=float(args.post_train_lambda_bg or post_cfg.get("lambda_bg", 1.0)),
            lambda_curve=float(args.post_train_lambda_curve or post_cfg.get("lambda_curve", 1.0)),
            lambda_curv=float(args.post_train_lambda_curv or post_cfg.get("lambda_curv", 0.05)),
            lora_blocks=int(args.post_train_lora_blocks or post_cfg.get("lora_blocks", 3)),
            lora_r=int(args.post_train_lora_r or post_cfg.get("lora_r", 8)),
            lora_alpha=int(args.post_train_lora_alpha or post_cfg.get("lora_alpha", 16)),
            lora_dropout=float(args.post_train_lora_dropout or post_cfg.get("lora_dropout", 0.05)),
            lora_use_mlp=bool(args.post_train_lora_use_mlp or post_cfg.get("lora_use_mlp", False)),
            method=str(post_cfg.get("method", "sam")),
            sam_rho=float(sam_rho),
            log_every=int(get_cfg(cfg, ("train", "log_every"), 10)),
            device=device,
            output_path=post_out,
            best_path=best_out,
        )
    if args.post_train_only and args.steps:
        logger.warning("Ignoring --steps when --post-train-only is set")


if __name__ == "__main__":
    main()
