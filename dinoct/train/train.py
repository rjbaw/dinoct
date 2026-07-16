from __future__ import annotations

import argparse
import csv
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import amp, nn, optim
import torch.nn.functional as F
import yaml
from torch.utils.data import Subset

from ..data import DataAugmentationDINO, MaskingGenerator, collate_data_and_cast, make_data_loader, make_dataset
from ..data.datasets import OCT
from ..layers import DINOHead
from ..loss import DINOLoss, GramLoss, KoLeoLoss, iBOTPatchLoss
from ..models import build_backbone
from ..utils import fix_random_seeds
from .core.schedules import cosine_schedule, linear_warmup_cosine_decay
from .denoise_aux import DenoiseHead, denoise_recon_loss
from .param_groups import fuse_params_groups, get_params_groups_with_decay_fsdp
from .post_train import run_post_training

REPO_ROOT = Path(__file__).resolve().parents[2]
_PKG_CONFIGS = Path(__file__).resolve().parents[1] / "configs"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "ssl_default_config.yaml"
if not DEFAULT_CONFIG.exists() and (_PKG_CONFIGS / "ssl_default_config.yaml").exists():
    DEFAULT_CONFIG = _PKG_CONFIGS / "ssl_default_config.yaml"
DEFAULT_TRAIN_CONFIG = REPO_ROOT / "configs" / "train" / "oct.yaml"
if not DEFAULT_TRAIN_CONFIG.exists() and (_PKG_CONFIGS / "train" / "oct.yaml").exists():
    DEFAULT_TRAIN_CONFIG = _PKG_CONFIGS / "train" / "oct.yaml"

logger = logging.getLogger("dinoct")


def _parse_dataset_path(dataset_str: str) -> tuple[str, dict[str, str]]:
    parts = dataset_str.split(":")
    tokens: dict[str, str] = {}
    for token in parts[1:]:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        tokens[key] = value
    return parts[0], tokens


def _format_dataset_path(name: str, tokens: dict[str, str]) -> str:
    rebuilt = [name] + [f"{key}={value}" for key, value in tokens.items()]
    return ":".join(rebuilt)


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


def configure_dataloader_sharing(cfg: dict[str, Any]) -> None:
    num_workers = int(get_cfg(cfg, ("train", "num_workers"), 4))
    if num_workers <= 0:
        return
    strategy = str(get_cfg(cfg, ("train", "sharing_strategy"), "file_system")).strip().lower()
    if strategy in {"", "none", "default"}:
        return
    current = torch.multiprocessing.get_sharing_strategy()
    if current != strategy:
        torch.multiprocessing.set_sharing_strategy(strategy)
        logger.info("set torch multiprocessing sharing_strategy=%s for dataloader workers", strategy)


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
    name, tokens = _parse_dataset_path(dataset_str)
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
    return _format_dataset_path(name, tokens) if tokens else dataset_str


def _global_mask_grid_size(global_size: int, patch_size: int) -> int:
    if patch_size <= 0:
        raise ValueError("student.patch_size must be positive")
    if global_size % patch_size != 0:
        raise ValueError("crops.global_crops_size must be divisible by student.patch_size")
    return global_size // patch_size


def build_dataloader(cfg: dict[str, Any]) -> tuple[torch.utils.data.DataLoader, int]:
    configure_dataloader_sharing(cfg)
    global_size = int(get_cfg(cfg, ("crops", "global_crops_size"), 224))
    local_size = int(get_cfg(cfg, ("crops", "local_crops_size"), 96))
    local_num = int(get_cfg(cfg, ("crops", "local_crops_number"), 8))
    patch_size = int(get_cfg(cfg, ("student", "patch_size"), 14))
    global_mask_grid = _global_mask_grid_size(global_size, patch_size)
    dataset_str = resolve_dataset_path(
        str(get_cfg(cfg, ("train", "dataset_path"), "OCT:root=data/oct:extra=data/oct/extra"))
    )
    ssl_split = str(get_cfg(cfg, ("train", "ssl_split"), "train")).strip().lower()

    augment = DataAugmentationDINO(
        get_cfg(cfg, ("crops", "global_crops_scale"), (0.32, 1.0)),
        get_cfg(cfg, ("crops", "local_crops_scale"), (0.05, 0.32)),
        local_num,
        global_crops_size=global_size,
        local_crops_size=local_size,
        gram_teacher_crops_size=get_cfg(cfg, ("crops", "gram_teacher_crops_size"), None),
        gram_teacher_no_distortions=bool(get_cfg(cfg, ("crops", "gram_teacher_no_distortions"), False)),
        teacher_no_color_jitter=bool(get_cfg(cfg, ("teacher", "teacher_no_color_jitter"), False)),
        local_crops_subset_of_global_crops=bool(
            get_cfg(cfg, ("crops", "local_crops_subset_of_global_crops"), False)
        ),
        patch_size=patch_size,
        share_color_jitter=bool(get_cfg(cfg, ("train", "share_color_jitter"), False)),
        horizontal_flips=bool(get_cfg(cfg, ("train", "horizontal_flips"), True)),
        solarize_p=float(get_cfg(cfg, ("train", "solarize_p"), 0.2)),
        solarize_threshold=float(get_cfg(cfg, ("train", "solarize_threshold"), 128)),
        gaussian_noise_std=float(get_cfg(cfg, ("train", "gaussian_noise_std"), 0.0)),
        gaussian_noise_p=float(get_cfg(cfg, ("train", "gaussian_noise_p"), 0.0)),
        gaussian_noise_student_only=bool(get_cfg(cfg, ("train", "gaussian_noise_student_only"), True)),
        aggressive_aug=bool(get_cfg(cfg, ("train", "aggressive_aug"), False)),
        aggressive_blur=bool(get_cfg(cfg, ("train", "aggressive_blur"), False)),
        aggressive_solarize=bool(get_cfg(cfg, ("train", "aggressive_solarize"), False)),
        aggressive_jitter=bool(get_cfg(cfg, ("train", "aggressive_jitter"), False)),
        aggressive_elastic=bool(get_cfg(cfg, ("train", "aggressive_elastic"), False)),
        aggressive_erasing=bool(get_cfg(cfg, ("train", "aggressive_erasing"), False)),
        aggressive_noise=bool(get_cfg(cfg, ("train", "aggressive_noise"), False)),
        erasing_p=float(get_cfg(cfg, ("train", "erasing_p"), 0.0)),
        erasing_scale_max=float(get_cfg(cfg, ("train", "erasing_scale_max"), 0.2)),
        elastic_alpha=float(get_cfg(cfg, ("train", "elastic_alpha"), 0.0)),
        elastic_sigma=float(get_cfg(cfg, ("train", "elastic_sigma"), 5.0)),
    )

    _agg_flags = {
        k: bool(get_cfg(cfg, ("train", k), False))
        for k in ("aggressive_aug", "aggressive_blur", "aggressive_solarize", "aggressive_jitter",
                  "aggressive_elastic", "aggressive_erasing", "aggressive_noise")
    }
    if any(_agg_flags.values()):
        logger.info("AGGRESSIVE AUG enabled: %s", {k: v for k, v in _agg_flags.items() if v})

    mask_gen = MaskingGenerator(
        input_size=(global_mask_grid, global_mask_grid),
        max_num_patches=int(0.5 * global_mask_grid**2),
    )
    n_tokens = global_mask_grid**2
    collate_fn = lambda batch: collate_data_and_cast(  # noqa: E731
        batch,
        mask_ratio_tuple=tuple(get_cfg(cfg, ("ibot", "mask_ratio_min_max"), (0.1, 0.5))),
        mask_probability=float(get_cfg(cfg, ("ibot", "mask_sample_probability"), 0.5)),
        n_tokens=n_tokens,
        mask_generator=mask_gen,
        dtype=torch.float32,
    )

    dataset = make_dataset(dataset_str=dataset_str, transform=augment, target_transform=lambda _: ())
    ssl_files: list[Path] | None = None
    base_oct = dataset if isinstance(dataset, OCT) else None
    # Use the dataset's REAL root (handles hub download/extract that rewrites root inside
    # make_dataset), NOT the original config string. Reconstructing from the config string
    # makes split-filtering and the leak preflight silently operate on a non-existent local
    # path on hub nodes -> false PASS.
    real_root = Path(base_oct.root) if base_oct is not None else None
    do_filter = base_oct is not None and ssl_split not in {"", "all", "none", "false", "off"}
    if do_filter:
        entries = base_oct._get_entries()
        if "split" not in (entries.dtype.names or ()):
            raise RuntimeError(
                f"ssl_split={ssl_split!r} requested but OCT entries carry no 'split' field; "
                f"cannot filter the SSL pool -> refusing (would train on the full, possibly leaked pool)."
            )
        keep_idx = [idx for idx, entry in enumerate(entries) if str(entry["split"]).lower() == ssl_split]
        if not keep_idx:
            raise RuntimeError(
                f"ssl_split={ssl_split!r} matched 0 of {len(entries)} entries; check the split value."
            )
        logger.info("using OCT split=%s entries for SSL (real root %s)", ssl_split, real_root)
        logger.info(
            "effective SSL subset after split filtering: %d / %d samples",
            len(keep_idx),
            len(entries),
        )
        ssl_files = [real_root / str(entries[i]["filename"]) for i in keep_idx]
        dataset = Subset(dataset, keep_idx)
    else:
        if base_oct is not None:
            ssl_files = [real_root / str(e["filename"]) for e in base_oct._get_entries()]
        try:
            logger.info("effective SSL dataset size without split filtering: %d samples", len(dataset))
        except TypeError:
            pass

    # Fail-closed leak preflight. Entries always carry split metadata for OCT, so splits are
    # "resolvable" and Guard 1 enforces ssl_split in the allowlist regardless of any LOCAL splits.csv.
    if bool(get_cfg(cfg, ("train", "leak_preflight"), True)) and base_oct is not None:
        from dinoct.data.leak_preflight import assert_no_ssl_leak

        splits_exist = "split" in (base_oct._get_entries().dtype.names or ())
        cfg_eval = get_cfg(cfg, ("train", "leak_eval_dirs"), ["data/oct/eval/hard"])
        eval_dirs: list[Path] = []
        for d in cfg_eval:
            d = Path(d)
            cand = (real_root / "eval" / d.name) if real_root is not None else None
            eval_dirs.append(cand if (cand is not None and cand.exists()) else d)
        allow_splits = tuple(
            str(s).strip().lower()
            for s in get_cfg(cfg, ("train", "leak_allow_splits"), ["train"])
        )
        assert_no_ssl_leak(
            ssl_files=ssl_files,
            eval_dirs=eval_dirs,
            ssl_split=ssl_split,
            splits_exist=splits_exist,
            allow_splits=allow_splits,
            deep_md5=bool(get_cfg(cfg, ("train", "leak_preflight_md5"), True)),
            logger=logger,
        )

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


def _cfg_path_is_set(path: Any) -> bool:
    return path is not None and str(path).strip().lower() not in {"", "none", "null"}


def _build_backbone_from_cfg(cfg: dict[str, Any], device: torch.device) -> nn.Module:
    arch_raw = str(get_cfg(cfg, ("student", "arch"), "vit_small"))
    arch = arch_raw.replace("vit_", "") if arch_raw.startswith("vit_") else arch_raw
    return build_backbone(
        arch,
        patch_size=int(get_cfg(cfg, ("student", "patch_size"), 14)),
        drop_path_rate=float(get_cfg(cfg, ("student", "drop_path_rate"), 0.0)),
        drop_path_uniform=bool(get_cfg(cfg, ("student", "drop_path_uniform"), False)),
        block_chunks=int(get_cfg(cfg, ("student", "block_chunks"), 0)),
        layerscale_init=get_cfg(cfg, ("student", "layerscale"), None),
        ffn_layer=_resolve_ffn_layer(str(get_cfg(cfg, ("student", "ffn_layer"), "mlp"))),
        qkv_bias=bool(get_cfg(cfg, ("student", "qkv_bias"), True)),
        proj_bias=bool(get_cfg(cfg, ("student", "proj_bias"), True)),
        ffn_bias=bool(get_cfg(cfg, ("student", "ffn_bias"), True)),
        n_storage_tokens=int(get_cfg(cfg, ("student", "n_storage_tokens"), 0)),
        mask_k_bias=bool(get_cfg(cfg, ("student", "mask_k_bias"), False)),
        device=device,
    ).to(device)


def _extract_backbone_state(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise ValueError("Gram checkpoint must be a state dict or a checkpoint dict")
    if isinstance(checkpoint.get("teacher"), dict):
        return checkpoint["teacher"]
    if isinstance(checkpoint.get("student"), dict):
        return checkpoint["student"]
    if isinstance(checkpoint.get("model"), dict):
        checkpoint = checkpoint["model"]

    prefixes = ("teacher.backbone.", "module.teacher.backbone.", "student.backbone.", "module.student.backbone.")
    for prefix in prefixes:
        state = {k[len(prefix) :]: v for k, v in checkpoint.items() if isinstance(k, str) and k.startswith(prefix)}
        if state:
            return state
    return checkpoint


def load_gram_teacher_checkpoint(gram_teacher: nn.Module, ckpt_path: Path | str, device: torch.device) -> None:
    checkpoint = torch.load(Path(ckpt_path), map_location=device)
    state = _extract_backbone_state(checkpoint)
    missing, unexpected = gram_teacher.load_state_dict(state, strict=False)
    if missing:
        logger.warning("Gram teacher checkpoint missing %d keys; first missing key: %s", len(missing), missing[0])
    if unexpected:
        logger.warning("Gram teacher checkpoint has %d unexpected keys; first unexpected key: %s", len(unexpected), unexpected[0])


def copy_gram_teacher_from_teacher(bundle: "ModelBundle") -> None:
    if bundle.gram_teacher is None:
        return
    bundle.gram_teacher.load_state_dict(bundle.teacher.state_dict(), strict=False)
    for p in bundle.gram_teacher.parameters():
        p.requires_grad = False
    bundle.gram_teacher.eval()
    bundle.gram_teacher_initialized = True


@dataclass
class ModelBundle:
    student: nn.Module
    teacher: nn.Module
    student_head: DINOHead
    teacher_head: DINOHead
    student_ibot_head: DINOHead | None
    teacher_ibot_head: DINOHead | None
    gram_teacher: nn.Module | None = None
    lepa_predictor: nn.Module | None = None
    denoise_head: nn.Module | None = None
    gram_teacher_initialized: bool = False


class LEPAPredictor(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_teacher_eval(bundle: ModelBundle) -> None:
    bundle.teacher.eval()
    bundle.teacher_head.eval()
    if bundle.teacher_ibot_head is not None:
        bundle.teacher_ibot_head.eval()
    if bundle.gram_teacher is not None:
        bundle.gram_teacher.eval()


def build_pretrain_checkpoint(
    bundle: ModelBundle,
    cfg: dict[str, Any],
    *,
    iteration: int,
    optimizer: optim.Optimizer | None = None,
    scaler: amp.GradScaler | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    checkpoint = {
        "student": bundle.student.state_dict(),
        "student_head": bundle.student_head.state_dict(),
        "student_ibot_head": bundle.student_ibot_head.state_dict() if bundle.student_ibot_head else None,
        "teacher": bundle.teacher.state_dict(),
        "teacher_head": bundle.teacher_head.state_dict(),
        "teacher_ibot_head": bundle.teacher_ibot_head.state_dict() if bundle.teacher_ibot_head else None,
        "gram_teacher": bundle.gram_teacher.state_dict() if bundle.gram_teacher else None,
        "lepa_predictor": bundle.lepa_predictor.state_dict() if bundle.lepa_predictor else None,
        "denoise_head": bundle.denoise_head.state_dict() if bundle.denoise_head else None,
        "iteration": int(iteration),
        "config": cfg,
    }
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()
    if extra:
        checkpoint.update(extra)
    return checkpoint


def save_pretrain_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    torch.save(checkpoint, tmp_path)
    tmp_path.replace(path)


def _unique_backup_path(path: Path, suffix: str) -> Path:
    backup = path.with_name(path.name + suffix)
    if not backup.exists():
        return backup
    idx = 1
    while True:
        candidate = path.with_name(f"{path.name}{suffix}.{idx}")
        if not candidate.exists():
            return candidate
        idx += 1


def _prepare_metrics_for_resume(metrics_path: Path, metric_header: list[str], resume_step: int) -> bool:
    if resume_step <= 0 or not metrics_path.exists() or metrics_path.stat().st_size == 0:
        return False

    raw_metrics = metrics_path.read_text()
    rows = list(csv.reader(raw_metrics.splitlines()))
    if not rows:
        return False

    if rows[0] and rows[0][0] == "step":
        header = rows[0]
        data_rows = rows[1:]
    else:
        header = metric_header
        data_rows = rows

    def align_row(row: list[str]) -> list[str]:
        source_header = metric_header if len(row) == len(metric_header) else header
        values = {
            key: row[idx]
            for idx, key in enumerate(source_header)
            if key and idx < len(row)
        }
        return [values.get(key, "") for key in metric_header]

    kept_rows = [metric_header]
    dropped_rows = 0
    rewritten_rows = header != metric_header
    max_kept_step = 0
    for row in data_rows:
        if not row:
            continue
        try:
            step = int(float(row[0]))
        except (TypeError, ValueError):
            dropped_rows += 1
            continue
        if step <= resume_step:
            aligned = align_row(row)
            kept_rows.append(aligned)
            rewritten_rows = rewritten_rows or aligned != row
            max_kept_step = max(max_kept_step, step)
        else:
            dropped_rows += 1

    if dropped_rows > 0 or rewritten_rows:
        backup_path = _unique_backup_path(metrics_path, ".before_resume_trim")
        backup_path.write_text(raw_metrics)
        with metrics_path.open("w", newline="") as fh:
            csv.writer(fh).writerows(kept_rows)
        logger.info(
            "prepared %s for resume at step %d: kept through step %d, dropped %d rows, backup=%s",
            metrics_path,
            resume_step,
            max_kept_step,
            dropped_rows,
            backup_path,
        )
    return True


def _load_module_checkpoint(
    module: nn.Module | None,
    checkpoint: dict[str, Any],
    key: str,
    *,
    optional_if_missing: bool = False,
) -> None:
    state = checkpoint.get(key)
    if module is None:
        if state is not None:
            raise ValueError(f"Checkpoint contains {key}, but the current config does not build it.")
        return
    if state is None:
        if optional_if_missing:
            logger.warning("Pretrain checkpoint is missing optional %s; keeping current initialization.", key)
            return
        raise ValueError(f"Checkpoint is missing {key}, but the current config requires it.")
    state = _convert_legacy_weight_norm_state(state)
    module.load_state_dict(state, strict=True)


def _convert_legacy_weight_norm_state(state: dict[str, Any]) -> dict[str, Any]:
    """Convert old parametrizations.weight keys to weight_norm's weight_g/v keys."""
    old_g = "last_layer.parametrizations.weight.original0"
    old_v = "last_layer.parametrizations.weight.original1"
    if old_g not in state and old_v not in state:
        return state
    converted = dict(state)
    if old_g in converted:
        converted["last_layer.weight_g"] = converted.pop(old_g)
    if old_v in converted:
        converted["last_layer.weight_v"] = converted.pop(old_v)
    return converted


def _move_optimizer_state_to_device(optimizer: optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def load_pretrain_checkpoint(
    path: Path,
    *,
    bundle: ModelBundle,
    optimizer: optim.Optimizer,
    scaler: amp.GradScaler,
    dino_loss: DINOLoss,
    ibot_loss: iBOTPatchLoss,
    device: torch.device,
) -> tuple[int, int]:
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Invalid pretrain checkpoint: {path}")

    _load_module_checkpoint(bundle.student, checkpoint, "student")
    _load_module_checkpoint(bundle.student_head, checkpoint, "student_head")
    _load_module_checkpoint(bundle.student_ibot_head, checkpoint, "student_ibot_head")
    _load_module_checkpoint(bundle.teacher, checkpoint, "teacher")
    _load_module_checkpoint(bundle.teacher_head, checkpoint, "teacher_head")
    _load_module_checkpoint(bundle.teacher_ibot_head, checkpoint, "teacher_ibot_head")
    _load_module_checkpoint(
        bundle.gram_teacher,
        checkpoint,
        "gram_teacher",
        optional_if_missing=bundle.gram_teacher is not None and bundle.gram_teacher_initialized,
    )
    _load_module_checkpoint(
        bundle.lepa_predictor,
        checkpoint,
        "lepa_predictor",
        optional_if_missing=bundle.lepa_predictor is not None,
    )
    _load_module_checkpoint(
        bundle.denoise_head,
        checkpoint,
        "denoise_head",
        optional_if_missing=bundle.denoise_head is not None,
    )
    if bundle.gram_teacher is not None and checkpoint.get("gram_teacher") is not None:
        bundle.gram_teacher_initialized = True
    set_teacher_eval(bundle)

    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        _move_optimizer_state_to_device(optimizer, device)
    else:
        logger.warning("Pretrain checkpoint has no optimizer state: %s", path)
    if "scaler" in checkpoint and checkpoint["scaler"] is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    if "dino_loss" in checkpoint:
        dino_loss.load_state_dict(checkpoint["dino_loss"])
    if "ibot_loss" in checkpoint:
        ibot_loss.load_state_dict(checkpoint["ibot_loss"])

    iteration = int(checkpoint.get("iteration", 0))
    gram_updates_done = int(checkpoint.get("gram_updates_done", 0))
    return iteration, gram_updates_done


def build_models(cfg: dict[str, Any], device: torch.device) -> ModelBundle:
    student = _build_backbone_from_cfg(cfg, device)
    teacher = _build_backbone_from_cfg(cfg, device)
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
    if not ibot_separate:
        if ibot_out != dino_out:
            raise ValueError(
                "ibot.head_n_prototypes must equal dino.head_n_prototypes when ibot.separate_head=false"
            )
        if (ibot_hidden, ibot_bottleneck, ibot_nlayers) != (dino_hidden, dino_bottleneck, dino_nlayers):
            raise ValueError(
                "iBOT head hidden_dim, bottleneck_dim, and nlayers must match DINO head settings "
                "when ibot.separate_head=false"
            )
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

    gram_teacher = None
    gram_teacher_initialized = False
    if bool(get_cfg(cfg, ("gram", "use_loss"), False)) and not bool(get_cfg(cfg, ("gram", "ema_teacher"), False)):
        gram_teacher = _build_backbone_from_cfg(cfg, device)
        for p in gram_teacher.parameters():
            p.requires_grad = False
        gram_ckpt = get_cfg(cfg, ("gram", "ckpt"), None)
        if _cfg_path_is_set(gram_ckpt):
            load_gram_teacher_checkpoint(gram_teacher, Path(str(gram_ckpt)), device)
            gram_teacher_initialized = True
        elif int(get_cfg(cfg, ("gram", "it_load_ema_teacher"), -1)) < 0:
            raise ValueError("Set gram.ckpt, gram.ema_teacher=true, or gram.it_load_ema_teacher>=0 to use Gram loss.")

    lepa_predictor = None
    if bool(get_cfg(cfg, ("lepa", "use_loss"), False)):
        lepa_predictor = LEPAPredictor(
            student.embed_dim,
            int(get_cfg(cfg, ("lepa", "predictor_hidden_dim"), max(1024, int(student.embed_dim) * 2))),
        ).to(device)

    denoise_head = None
    if bool(get_cfg(cfg, ("train", "denoise_aux"), False)):
        denoise_head = DenoiseHead(
            int(student.embed_dim),
            out_size=int(get_cfg(cfg, ("crops", "global_crops_size"), 224)),
            mid=int(get_cfg(cfg, ("train", "denoise_head_mid"), 256)),
        ).to(device)

    bundle = ModelBundle(
        student=student,
        teacher=teacher,
        student_head=student_head,
        teacher_head=teacher_head,
        student_ibot_head=student_ibot_head,
        teacher_ibot_head=teacher_ibot_head,
        gram_teacher=gram_teacher,
        lepa_predictor=lepa_predictor,
        denoise_head=denoise_head,
        gram_teacher_initialized=gram_teacher_initialized,
    )
    set_teacher_eval(bundle)
    return bundle


@dataclass
class Schedules:
    lr: list[float]
    weight_decay: list[float]
    teacher_temp: list[float]
    momentum: list[float]
    dino_loss_weight: list[float]
    ibot_loss_weight: list[float]
    gram_loss_weight: list[float]
    lepa_loss_weight: list[float]
    freeze_last_layer_iters: int


def _build_loss_weight_schedule(
    cfg: dict[str, Any],
    section: str,
    *,
    default: float,
    epoch_length: int,
    total_iters: int,
) -> list[float]:
    schedule_cfg = get_cfg(cfg, (section, "loss_weight_schedule"), None)
    base_weight = float(get_cfg(cfg, (section, "loss_weight"), default))
    if not isinstance(schedule_cfg, dict):
        return [base_weight] * total_iters

    warmup_iters = int(float(schedule_cfg.get("warmup_epochs", 0)) * epoch_length)
    cosine_iters = (
        int(float(schedule_cfg["cosine_epochs"]) * epoch_length)
        if "cosine_epochs" in schedule_cfg
        else None
    )
    return linear_warmup_cosine_decay(
        start=float(schedule_cfg.get("start", base_weight)),
        peak=float(schedule_cfg.get("peak", base_weight)),
        end=float(schedule_cfg.get("end", base_weight)),
        warmup_iterations=warmup_iters,
        total_iterations=total_iters,
        cosine_iterations=cosine_iters,
    )


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
    dino_loss_schedule = _build_loss_weight_schedule(
        cfg,
        "dino",
        default=1.0,
        epoch_length=epoch_length,
        total_iters=total_iters,
    )
    ibot_loss_schedule = _build_loss_weight_schedule(
        cfg,
        "ibot",
        default=1.0,
        epoch_length=epoch_length,
        total_iters=total_iters,
    )
    gram_loss_schedule = [0.0] * total_iters
    if bool(get_cfg(cfg, ("gram", "use_loss"), False)):
        gram_schedule_cfg = get_cfg(cfg, ("gram", "loss_weight_schedule"), None)
        if isinstance(gram_schedule_cfg, dict):
            gram_warmup_iters = int(float(gram_schedule_cfg.get("warmup_epochs", 0)) * epoch_length)
            gram_cosine_iters = (
                int(float(gram_schedule_cfg["cosine_epochs"]) * epoch_length)
                if "cosine_epochs" in gram_schedule_cfg
                else None
            )
            gram_loss_schedule = linear_warmup_cosine_decay(
                start=float(gram_schedule_cfg.get("start", 0.0)),
                peak=float(gram_schedule_cfg.get("peak", get_cfg(cfg, ("gram", "loss_weight"), 1.0))),
                end=float(gram_schedule_cfg.get("end", get_cfg(cfg, ("gram", "loss_weight"), 1.0))),
                warmup_iterations=gram_warmup_iters,
                total_iterations=total_iters,
                cosine_iterations=gram_cosine_iters,
            )
        else:
            gram_loss_schedule = [float(get_cfg(cfg, ("gram", "loss_weight"), 1.0))] * total_iters

    lepa_loss_schedule = [0.0] * total_iters
    if bool(get_cfg(cfg, ("lepa", "use_loss"), False)):
        lepa_schedule_cfg = get_cfg(cfg, ("lepa", "loss_weight_schedule"), None)
        if isinstance(lepa_schedule_cfg, dict):
            lepa_warmup_iters = int(float(lepa_schedule_cfg.get("warmup_epochs", 0)) * epoch_length)
            lepa_cosine_iters = (
                int(float(lepa_schedule_cfg["cosine_epochs"]) * epoch_length)
                if "cosine_epochs" in lepa_schedule_cfg
                else None
            )
            lepa_loss_schedule = linear_warmup_cosine_decay(
                start=float(lepa_schedule_cfg.get("start", 0.0)),
                peak=float(lepa_schedule_cfg.get("peak", get_cfg(cfg, ("lepa", "loss_weight"), 0.0))),
                end=float(lepa_schedule_cfg.get("end", get_cfg(cfg, ("lepa", "loss_weight"), 0.0))),
                warmup_iterations=lepa_warmup_iters,
                total_iterations=total_iters,
                cosine_iterations=lepa_cosine_iters,
            )
        else:
            lepa_loss_schedule = [float(get_cfg(cfg, ("lepa", "loss_weight"), 0.0))] * total_iters

    backbone_param_groups = get_params_groups_with_decay_fsdp(
        bundle.student,
        layerwise_decay=float(get_cfg(cfg, ("optim", "layerwise_decay"), 1.0)),
        patch_embed_lr_mult=float(get_cfg(cfg, ("optim", "patch_embed_lr_mult"), 1.0)),
        wd_grouping=str(get_cfg(cfg, ("optim", "wd_grouping"), "current")),
    )
    param_groups = fuse_params_groups(backbone_param_groups)

    head_groups = [
        {"params": bundle.student_head.parameters(), "weight_decay": 1.0, "lr_multiplier": 1.0},
    ]
    if bundle.student_ibot_head is not None:
        head_groups.append({"params": bundle.student_ibot_head.parameters(), "weight_decay": 1.0, "lr_multiplier": 1.0})
    param_groups.extend(head_groups)
    if bundle.lepa_predictor is not None:
        param_groups.append(
            {
                "params": bundle.lepa_predictor.parameters(),
                "weight_decay": float(get_cfg(cfg, ("lepa", "predictor_weight_decay"), 1.0)),
                "lr_multiplier": float(get_cfg(cfg, ("lepa", "predictor_lr_mult"), 1.0)),
            }
        )
    if bundle.denoise_head is not None:
        param_groups.append(
            {
                "params": bundle.denoise_head.parameters(),
                "weight_decay": float(get_cfg(cfg, ("train", "denoise_head_weight_decay"), 0.0)),
                "lr_multiplier": float(get_cfg(cfg, ("train", "denoise_head_lr_mult"), 1.0)),
            }
        )

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
        dino_loss_weight=dino_loss_schedule,
        ibot_loss_weight=ibot_loss_schedule,
        gram_loss_weight=gram_loss_schedule,
        lepa_loss_weight=lepa_loss_schedule,
        freeze_last_layer_iters=freeze_epochs * epoch_length,
    )
    return optimizer, schedules


def update_teacher_weights(student: nn.Module, teacher: nn.Module, momentum: float) -> None:
    with torch.no_grad():
        for ps, pt in zip(student.parameters(), teacher.parameters()):
            pt.data.mul_(momentum).add_(ps.data, alpha=1.0 - momentum)


def maybe_refresh_gram_teacher(
    cfg: dict[str, Any],
    bundle: ModelBundle,
    iteration: int,
    updates_done: int,
) -> int:
    if bundle.gram_teacher is None:
        return updates_done

    it_load = int(get_cfg(cfg, ("gram", "it_load_ema_teacher"), -1))
    if not bundle.gram_teacher_initialized and it_load >= 0 and iteration >= it_load:
        copy_gram_teacher_from_teacher(bundle)
        logger.info("initialized Gram teacher from EMA teacher at iteration %d", iteration + 1)

    if not bool(get_cfg(cfg, ("gram", "rep_update"), False)) or not bundle.gram_teacher_initialized:
        return updates_done

    max_updates = get_cfg(cfg, ("gram", "max_updates"), None)
    if max_updates is not None and updates_done >= int(max_updates):
        return updates_done

    first_update = int(get_cfg(cfg, ("gram", "it_first_update"), 0))
    update_frequency = max(1, int(get_cfg(cfg, ("gram", "update_frequency"), 50000)))
    if iteration >= first_update and (iteration - first_update) % update_frequency == 0:
        copy_gram_teacher_from_teacher(bundle)
        updates_done += 1
        logger.info("updated Gram teacher from EMA teacher at iteration %d", iteration + 1)
    return updates_done


def _resize_patch_tokens_to_match(
    source: torch.Tensor,
    target_num_patches: int,
    *,
    mode: str,
    antialias: bool,
) -> torch.Tensor:
    if source.shape[1] == target_num_patches:
        return source
    source_side = math.isqrt(source.shape[1])
    target_side = math.isqrt(target_num_patches)
    if source_side * source_side != source.shape[1] or target_side * target_side != target_num_patches:
        raise ValueError(
            f"Cannot resize non-square patch tokens from {source.shape[1]} to {target_num_patches} patches"
        )
    source_grid = source.transpose(1, 2).reshape(source.shape[0], source.shape[2], source_side, source_side)
    align_corners = False if mode in {"linear", "bilinear", "bicubic", "trilinear"} else None
    resized = F.interpolate(
        source_grid,
        size=(target_side, target_side),
        mode=mode,
        align_corners=align_corners,
        antialias=antialias if mode in {"bilinear", "bicubic"} else False,
    )
    return resized.flatten(2).transpose(1, 2)


def _select_gram_tokens(
    student_tokens: torch.Tensor,
    teacher_tokens: torch.Tensor,
    masks: torch.Tensor,
    tokens_used: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tokens_used == "all":
        return student_tokens, teacher_tokens
    if tokens_used == "masked":
        selection = masks
    elif tokens_used == "unmasked":
        selection = ~masks
    else:
        raise ValueError("gram.tokens_used must be one of: all, masked, unmasked")
    return student_tokens[selection], teacher_tokens[selection]


def _lepa_prediction_loss(
    predictor: nn.Module,
    student_tokens: torch.Tensor,
    teacher_tokens: torch.Tensor,
    masks: torch.Tensor,
    mask_indices_list: torch.Tensor,
    masks_weight: torch.Tensor,
    *,
    tokens_used: str,
    normalized: bool,
    loss_type: str,
    smooth_l1_beta: float,
) -> torch.Tensor:
    student_flat = student_tokens.flatten(0, 1)
    teacher_flat = teacher_tokens.detach().flatten(0, 1)
    if tokens_used == "masked":
        if mask_indices_list.numel() == 0:
            return student_flat.new_zeros(())
        student_selected = student_flat.index_select(dim=0, index=mask_indices_list)
        teacher_selected = teacher_flat.index_select(dim=0, index=mask_indices_list)
        weights = masks_weight.to(device=student_selected.device, dtype=torch.float32)
    elif tokens_used in {"all", "unmasked"}:
        selection = torch.ones_like(masks, dtype=torch.bool) if tokens_used == "all" else ~masks.bool()
        if not bool(selection.any()):
            return student_flat.new_zeros(())
        student_selected = student_tokens[selection]
        teacher_selected = teacher_tokens.detach()[selection]
        weights = None
    else:
        raise ValueError("lepa.tokens_used must be one of: all, masked, unmasked")

    pred = predictor(student_selected)
    pred_f = pred.float()
    target_f = teacher_selected.float()
    if normalized:
        pred_f = F.normalize(pred_f, dim=-1)
        target_f = F.normalize(target_f, dim=-1)

    loss_type = loss_type.lower()
    if loss_type == "smooth_l1":
        per_token = F.smooth_l1_loss(pred_f, target_f, reduction="none", beta=float(smooth_l1_beta)).sum(dim=-1)
    elif loss_type == "mse":
        per_token = F.mse_loss(pred_f, target_f, reduction="none").sum(dim=-1)
    elif loss_type == "cosine":
        per_token = 1.0 - F.cosine_similarity(pred_f, target_f, dim=-1)
    else:
        raise ValueError("lepa.loss_type must be one of: smooth_l1, mse, cosine")

    if weights is not None:
        return (per_token * weights).sum() / max(int(masks.shape[0]), 1)
    return per_token.mean()


def apply_freeze_last_layer(head: DINOHead, iteration: int, freeze_until: int) -> None:
    requires_grad = iteration >= freeze_until
    for name, p in head.last_layer.named_parameters():
        if name.endswith("weight_g"):
            p.requires_grad = False
        else:
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
        # backend=None must not be passed through: torch.compile treats an explicit None
        # as the compiler fn and fails lazily at the first forward ('NoneType' not callable).
        if backend:
            return torch.compile(module, backend=backend)
        return torch.compile(module)
    except Exception as exc:  # pragma: no cover - best-effort compile
        logger.warning("torch.compile failed (%s); continuing without compilation", exc)
        return module


@torch.no_grad()
def _probability_stats(probs: torch.Tensor | None) -> tuple[float, float, float]:
    """Return mean entropy, prototype-usage entropy, and mean max probability."""
    if probs is None or probs.numel() == 0:
        return math.nan, math.nan, math.nan
    probs = probs.detach().float().reshape(-1, probs.shape[-1])
    n_classes = probs.shape[-1]
    if n_classes <= 1:
        return math.nan, math.nan, math.nan
    log_k = math.log(n_classes)
    safe_probs = probs.clamp_min(1e-12)
    sample_entropy = -(safe_probs * safe_probs.log()).sum(dim=-1).mean() / log_k
    usage = probs.mean(dim=0).clamp_min(1e-12)
    usage_entropy = -(usage * usage.log()).sum() / log_k
    mean_max_prob = probs.max(dim=-1).values.mean()
    return float(sample_entropy), float(usage_entropy), float(mean_max_prob)


def train(
    cfg: dict[str, Any],
    *,
    steps_override: int | None = None,
    schedule_steps_override: int | None = None,
    output_dir_override: Path | None = None,
    seed_override: int | None = None,
    resume_pretrain: Path | None = None,
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
    configured_total_iters = epochs * epoch_length
    total_iters = steps_override if steps_override is not None else configured_total_iters
    schedule_iters = schedule_steps_override if schedule_steps_override is not None else total_iters
    if schedule_iters < total_iters:
        raise ValueError(
            f"--schedule-steps ({schedule_iters}) must be >= run steps ({total_iters}); "
            "otherwise schedule arrays would end before training."
        )

    optimizer, schedules = build_optimizer_and_schedules(cfg, bundle, schedule_iters)

    dino_loss = DINOLoss(
        out_dim=int(get_cfg(cfg, ("dino", "head_n_prototypes"), 65536)),
        student_temp=float(get_cfg(cfg, ("dino", "student_temp"), 0.1)),
        sinkhorn_queue_size=int(get_cfg(cfg, ("dino", "sinkhorn_queue_size"), 0)),
        sinkhorn_queue_start_iter=int(get_cfg(cfg, ("dino", "sinkhorn_queue_start_iter"), 0)),
    ).to(device)
    ibot_loss = iBOTPatchLoss(
        patch_out_dim=int(
            get_cfg(cfg, ("ibot", "head_n_prototypes"), get_cfg(cfg, ("dino", "head_n_prototypes"), 65536))
        ),
        student_temp=float(get_cfg(cfg, ("ibot", "student_temp"), 0.1)),
    ).to(device)
    koleo_loss = KoLeoLoss().to(device)
    gram_use_loss = bool(get_cfg(cfg, ("gram", "use_loss"), False))
    gram_img_level = bool(get_cfg(cfg, ("gram", "img_level"), False))
    gram_tokens_used = str(get_cfg(cfg, ("gram", "tokens_used"), "all")).lower()

    # Denoising/robustness aux: additive, ramped recon loss. Off unless enabled.
    denoise_weight = float(get_cfg(cfg, ("train", "denoise_weight"), 0.0)) if bundle.denoise_head is not None else 0.0
    denoise_warmup_iters = int(get_cfg(cfg, ("train", "denoise_warmup_iters"), 10000))
    denoise_corrupt_kwargs = {
        "noise_std": float(get_cfg(cfg, ("train", "denoise_noise_std"), 0.4)),
        "dropout_p": float(get_cfg(cfg, ("train", "denoise_dropout_p"), 0.5)),
        "dropout_frac": float(get_cfg(cfg, ("train", "denoise_dropout_frac"), 0.25)),
        "intensity": float(get_cfg(cfg, ("train", "denoise_intensity"), 0.15)),
    }
    if denoise_weight > 0.0:
        logger.info(
            "denoise aux ENABLED: weight=%.3f warmup_iters=%d corrupt=%s",
            denoise_weight, denoise_warmup_iters, denoise_corrupt_kwargs,
        )
    if gram_img_level and gram_tokens_used != "all":
        raise ValueError("gram.tokens_used=masked/unmasked is only supported with gram.img_level=false.")
    gram_loss = (
        GramLoss(
            apply_norm=bool(get_cfg(cfg, ("gram", "normalized"), True)),
            remove_neg=bool(get_cfg(cfg, ("gram", "remove_neg"), False)),
            remove_only_teacher_neg=bool(get_cfg(cfg, ("gram", "remove_only_teacher_neg"), False)),
        ).to(device)
        if gram_use_loss
        else None
    )
    gram_ema_teacher = bool(get_cfg(cfg, ("gram", "ema_teacher"), False))
    if gram_ema_teacher and _cfg_path_is_set(get_cfg(cfg, ("gram", "ckpt"), None)):
        raise ValueError("Cannot use both gram.ema_teacher=true and gram.ckpt.")
    if gram_ema_teacher and get_cfg(cfg, ("crops", "gram_teacher_crops_size"), None) is not None:
        raise ValueError("crops.gram_teacher_crops_size must be null when gram.ema_teacher=true.")

    scaler = amp.GradScaler("cuda", enabled=bool(get_cfg(cfg, ("compute_precision", "grad_scaler"), True)))
    centering = str(get_cfg(cfg, ("train", "centering"), "centering")).lower()
    output_dir = (
        Path(output_dir_override) if output_dir_override else Path(get_cfg(cfg, ("train", "output_dir"), "outputs"))
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    pretrain_dir = output_dir / "pretrain"
    pretrain_dir.mkdir(parents=True, exist_ok=True)
    setup_file_logging(pretrain_dir / "train.log")
    start_iteration = 0
    gram_updates_done = 0
    if resume_pretrain is not None:
        start_iteration, gram_updates_done = load_pretrain_checkpoint(
            resume_pretrain,
            bundle=bundle,
            optimizer=optimizer,
            scaler=scaler,
            dino_loss=dino_loss,
            ibot_loss=ibot_loss,
            device=device,
        )
        if start_iteration > total_iters:
            raise ValueError(
                f"Resume checkpoint iteration {start_iteration} exceeds requested total iterations {total_iters}."
            )
        logger.info("resumed pretrain checkpoint %s at iteration %d", resume_pretrain, start_iteration)

    metric_header = [
        "step",
        "loss",
        "core_loss",
        "aux_loss",
        "dino",
        "ibot",
        "gram",
        "lepa",
        "koleo",
        "dino_contrib",
        "ibot_contrib",
        "gram_contrib",
        "lepa_contrib",
        "koleo_contrib",
        "dino_weight",
        "ibot_weight",
        "gram_weight",
        "lepa_weight",
        "koleo_weight",
        "grad_norm",
        "step_skipped",
        "lr",
        "weight_decay",
        "teacher_temp",
        "momentum",
        "dino_teacher_entropy",
        "dino_teacher_usage_entropy",
        "dino_teacher_max_prob",
        "dino_student_entropy",
        "dino_student_usage_entropy",
        "dino_student_max_prob",
    ]
    metrics_path = pretrain_dir / "metrics.csv"
    if start_iteration > 0:
        _prepare_metrics_for_resume(metrics_path, metric_header, start_iteration)
    append_metrics = start_iteration > 0 and metrics_path.exists() and metrics_path.stat().st_size > 0
    metrics_fh = metrics_path.open("a" if append_metrics else "w", newline="")
    metrics_writer = csv.writer(metrics_fh)
    if not append_metrics:
        metrics_writer.writerow(metric_header)

    logger.info(
        "starting DINOv3 training: %d iterations (~%d epochs), schedules span %d iterations, start iteration %d",
        total_iters,
        total_iters // epoch_length,
        schedule_iters,
        start_iteration,
    )
    if start_iteration >= total_iters:
        metrics_fh.close()
        logger.info("resume checkpoint already reaches requested training length; no pretrain steps to run.")
        return
    saveckp_freq = int(get_cfg(cfg, ("train", "saveckp_freq"), 0))
    checkpoint_period = saveckp_freq * epoch_length if saveckp_freq > 0 else 0
    iterator = iter(data_loader)
    if start_iteration > 0:
        skip_batches = start_iteration % max(1, len(data_loader))
        for _ in range(skip_batches):
            try:
                next(iterator)
            except StopIteration:
                iterator = iter(data_loader)
                next(iterator)
        logger.info("advanced data iterator by %d batches after resume", skip_batches)

    for iteration in range(start_iteration, total_iters):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(data_loader)
            batch = next(iterator)

        lr = schedules.lr[iteration]
        weight_decay = schedules.weight_decay[iteration]
        momentum = schedules.momentum[iteration]
        teacher_temp = schedules.teacher_temp[iteration]
        dino_weight = schedules.dino_loss_weight[iteration]
        ibot_weight = schedules.ibot_loss_weight[iteration]
        gram_weight = schedules.gram_loss_weight[iteration]
        lepa_weight = schedules.lepa_loss_weight[iteration]
        if gram_use_loss:
            gram_updates_done = maybe_refresh_gram_teacher(cfg, bundle, iteration, gram_updates_done)

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
        teacher_global_crops = batch.get("collated_global_crops_teacher", batch["collated_global_crops"])
        teacher_global_list = list(teacher_global_crops.to(device, non_blocking=True).chunk(2))
        local_list = list(local_crops.chunk(int(get_cfg(cfg, ("crops", "local_crops_number"), 8))))
        global_masks = list(masks.chunk(2))
        mask_list = global_masks + [None for _ in local_list]

        with torch.no_grad():
            teacher_outputs = bundle.teacher.forward_features_list(
                teacher_global_list,
                [None for _ in teacher_global_list],
            )
            teacher_cls = [o["x_norm_clstoken"] for o in teacher_outputs]
            teacher_patches = [o["x_norm_patchtokens"] for o in teacher_outputs]
            teacher_logits = [bundle.teacher_head(t) for t in teacher_cls]
            teacher_concat = torch.cat(teacher_logits, dim=0)
            if centering == "sinkhorn_knopp":
                teacher_targets = dino_loss.sinkhorn_knopp_teacher(teacher_concat, teacher_temp, iteration=iteration)
            else:
                teacher_targets = dino_loss.softmax_center_teacher(teacher_concat, teacher_temp)
            (
                dino_teacher_entropy,
                dino_teacher_usage_entropy,
                dino_teacher_max_prob,
            ) = _probability_stats(teacher_targets)
            if torch.isfinite(teacher_concat).all():
                dino_loss.update_center(teacher_concat)
            else:
                # NaN guard: a single non-finite teacher batch would permanently
                # poison the center EMA buffer — drop it instead.
                logger.warning("non-finite teacher logits at iteration %d; DINO center update skipped", iteration)
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
                if torch.isfinite(teacher_masked_logits).all():
                    ibot_loss.update_center(teacher_masked_logits.unsqueeze(0))
                else:
                    logger.warning("non-finite iBOT teacher logits at iteration %d; center update skipped", iteration)
            else:
                teacher_ibot_targets = None

            gram_teacher_patches = None
            if gram_use_loss and gram_weight > 0:
                if gram_ema_teacher:
                    gram_teacher_patches = teacher_patches
                else:
                    if bundle.gram_teacher is None or not bundle.gram_teacher_initialized:
                        raise RuntimeError("Gram loss is active, but the Gram teacher has not been initialized.")
                    gram_crops = batch.get("collated_gram_teacher_crops")
                    if gram_crops is not None:
                        gram_list = list(gram_crops.to(device, non_blocking=True).chunk(2))
                    else:
                        gram_list = global_list
                    gram_outputs = bundle.gram_teacher.forward_features_list(gram_list, [None for _ in gram_list])
                    gram_teacher_patches = [o["x_norm_patchtokens"] for o in gram_outputs]

        with amp.autocast(device_type="cuda", enabled=scaler.is_enabled()):
            student_outputs = bundle.student.forward_features_list(global_list + local_list, mask_list)
            student_cls = [o["x_norm_clstoken"] for o in student_outputs]
            student_patches = [o["x_norm_patchtokens"] for o in student_outputs[:2]]  # mask applied only to globals

            student_logits = [bundle.student_head(t) for t in student_cls]
            teacher_targets = list(teacher_targets_split)
            student_global_logits = student_logits[:2]
            student_local_logits = student_logits[2:]
            with torch.no_grad():
                student_global_probs = F.softmax(
                    torch.cat([t.detach().float() for t in student_global_logits], dim=0) / dino_loss.student_temp,
                    dim=-1,
                )
                (
                    dino_student_entropy,
                    dino_student_usage_entropy,
                    dino_student_max_prob,
                ) = _probability_stats(student_global_probs)
            n_global_loss_terms = len(student_global_logits) * len(teacher_targets) - min(
                len(student_global_logits), len(teacher_targets)
            )
            n_local_loss_terms = len(student_local_logits) * len(teacher_targets)
            dino_global = dino_loss(student_global_logits, teacher_targets, ignore_diagonal=True)
            dino_local = (
                dino_loss(student_local_logits, teacher_targets)
                if student_local_logits
                else torch.tensor(0.0, device=device)
            )
            dino_term = (dino_global * n_global_loss_terms + dino_local * n_local_loss_terms) / (
                n_global_loss_terms + n_local_loss_terms
            )
            dino_contrib = dino_weight * dino_term
            loss = dino_contrib

            koleo_weight = float(get_cfg(cfg, ("dino", "koleo_loss_weight"), 0.0))
            # Two-stage KoLeo: keep koleo ON early (build feature-space spread) then turn it OFF at
            # koleo_off_iter (gain stability once the embedding has matured). koleo_off_iter<=0
            # => always on (backward-compatible).
            koleo_off_iter = int(get_cfg(cfg, ("dino", "koleo_off_iter"), 0))
            if koleo_off_iter > 0 and iteration >= koleo_off_iter:
                koleo_weight = 0.0
            koleo_term = torch.tensor(0.0, device=device)
            koleo_contrib = torch.tensor(0.0, device=device)
            if koleo_weight > 0:
                n_global = len(student_global_logits)
                koleo_term = sum(koleo_loss(t) for t in student_cls[:n_global]) / n_global
                koleo_contrib = koleo_weight * n_global * koleo_term
                loss = loss + koleo_contrib

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
                ibot_contrib = ibot_weight * ibot_term
                loss = loss + ibot_contrib
            else:
                ibot_contrib = torch.tensor(0.0, device=device)

            gram_term = torch.tensor(0.0, device=device)
            gram_contrib = torch.tensor(0.0, device=device)
            if gram_loss is not None and gram_weight > 0:
                if gram_teacher_patches is None:
                    raise RuntimeError("Gram teacher features were not computed for an active Gram loss.")
                gram_student = torch.cat(student_patches, dim=0)
                gram_teacher = torch.cat(gram_teacher_patches, dim=0)
                gram_teacher = _resize_patch_tokens_to_match(
                    gram_teacher,
                    gram_student.shape[1],
                    mode=str(get_cfg(cfg, ("gram", "global_teacher_resize_method"), "bicubic")),
                    antialias=bool(get_cfg(cfg, ("gram", "global_teacher_resize_antialias"), False)),
                )
                gram_student, gram_teacher = _select_gram_tokens(
                    gram_student,
                    gram_teacher,
                    masks,
                    gram_tokens_used,
                )
                if gram_student.numel() > 0:
                    gram_term = gram_loss(gram_student, gram_teacher, img_level=gram_img_level)
                    gram_contrib = gram_weight * gram_term
                    loss = loss + gram_contrib

            lepa_term = torch.tensor(0.0, device=device)
            lepa_contrib = torch.tensor(0.0, device=device)
            if bundle.lepa_predictor is not None and lepa_weight > 0:
                lepa_student = torch.cat(student_patches, dim=0)
                lepa_teacher = torch.cat(teacher_patches, dim=0)
                lepa_term = _lepa_prediction_loss(
                    bundle.lepa_predictor,
                    lepa_student,
                    lepa_teacher,
                    masks,
                    mask_indices_list,
                    masks_weight,
                    tokens_used=str(get_cfg(cfg, ("lepa", "tokens_used"), "masked")).lower(),
                    normalized=bool(get_cfg(cfg, ("lepa", "normalized"), True)),
                    loss_type=str(get_cfg(cfg, ("lepa", "loss_type"), "smooth_l1")),
                    smooth_l1_beta=float(get_cfg(cfg, ("lepa", "smooth_l1_beta"), 1.0)),
                )
                lepa_contrib = lepa_weight * lepa_term
                loss = loss + lepa_contrib

            denoise_term = torch.tensor(0.0, device=device)
            denoise_contrib = torch.tensor(0.0, device=device)
            if bundle.denoise_head is not None and denoise_weight > 0.0:
                dn_ramp = min(1.0, iteration / max(1, denoise_warmup_iters))
                denoise_term = denoise_recon_loss(
                    bundle.student, bundle.denoise_head, global_list, denoise_corrupt_kwargs
                )
                denoise_contrib = (denoise_weight * dn_ramp) * denoise_term
                loss = loss + denoise_contrib

            core_loss = dino_contrib + ibot_contrib
            aux_loss = gram_contrib + lepa_contrib + koleo_contrib + denoise_contrib

        optimizer.zero_grad(set_to_none=True)
        grad_norm_value = math.nan
        if not bool(torch.isfinite(loss.detach())):
            # NaN guard: never backward a non-finite loss — protects the weights AND
            # keeps the GradScaler scale from collapsing to its floor, where step_skipped
            # detection (get_scale() < old_scale) silently goes blind.
            logger.warning("non-finite loss at iteration %d; optimizer step skipped", iteration)
            step_skipped = True
        else:
            scaler.scale(loss).backward()
            clip_grad = float(get_cfg(cfg, ("optim", "clip_grad"), 0.0))
            if clip_grad > 0:
                scaler.unscale_(optimizer)
                params_to_clip = [p for group in optimizer.param_groups for p in group["params"] if p.requires_grad]
                grad_norm = nn.utils.clip_grad_norm_(params_to_clip, max_norm=clip_grad)
                grad_norm_value = float(grad_norm.detach().cpu()) if torch.is_tensor(grad_norm) else float(grad_norm)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            step_skipped = scaler.is_enabled() and scaler.get_scale() < old_scale
        if not step_skipped:
            update_teacher_weights(bundle.student, bundle.teacher, momentum)
            update_teacher_weights(bundle.student_head, bundle.teacher_head, momentum)
            if bundle.student_ibot_head is not None and bundle.teacher_ibot_head is not None:
                update_teacher_weights(bundle.student_ibot_head, bundle.teacher_ibot_head, momentum)

        if (iteration + 1) % max(1, int(get_cfg(cfg, ("train", "log_every"), 10))) == 0 or iteration == 0:
            metrics_writer.writerow(
                [
                    iteration + 1,
                    float(loss.detach()),
                    float(core_loss.detach()),
                    float(aux_loss.detach()),
                    float(dino_term.detach()),
                    float(ibot_term.detach()),
                    float(gram_term.detach()),
                    float(lepa_term.detach()),
                    float(koleo_term.detach()),
                    float(dino_contrib.detach()),
                    float(ibot_contrib.detach()),
                    float(gram_contrib.detach()),
                    float(lepa_contrib.detach()),
                    float(koleo_contrib.detach()),
                    dino_weight,
                    ibot_weight,
                    gram_weight,
                    lepa_weight,
                    koleo_weight,
                    grad_norm_value,
                    int(bool(step_skipped)),
                    lr,
                    weight_decay,
                    teacher_temp,
                    momentum,
                    dino_teacher_entropy,
                    dino_teacher_usage_entropy,
                    dino_teacher_max_prob,
                    dino_student_entropy,
                    dino_student_usage_entropy,
                    dino_student_max_prob,
                ]
            )
            metrics_fh.flush()
            logger.info(
                "[%05d/%05d] loss=%.4f core=%.4f aux=%.4f dino=%.4f ibot=%.4f gram=%.4f lepa=%.4f koleo=%.4f "
                "denoise=%.4f grad=%.3f lr=%.6f wd=%.4f temp=%.3f mom=%.4f",
                iteration + 1,
                total_iters,
                float(loss.detach()),
                float(core_loss.detach()),
                float(aux_loss.detach()),
                float(dino_term.detach()),
                float(ibot_term.detach()),
                float(gram_term.detach()),
                float(lepa_term.detach()),
                float(koleo_term.detach()),
                float(denoise_term.detach()),
                grad_norm_value,
                lr,
                weight_decay,
                teacher_temp,
                momentum,
            )

        step = iteration + 1
        if checkpoint_period > 0 and step % checkpoint_period == 0 and step < total_iters:
            checkpoint_path = pretrain_dir / f"checkpoint_{step:07d}.pth"
            save_pretrain_checkpoint(
                checkpoint_path,
                build_pretrain_checkpoint(
                    bundle,
                    cfg,
                    iteration=step,
                    optimizer=optimizer,
                    scaler=scaler,
                    extra={
                        "dino_loss": dino_loss.state_dict(),
                        "ibot_loss": ibot_loss.state_dict(),
                        "gram_updates_done": gram_updates_done,
                    },
                ),
            )
            logger.info("periodic checkpoint saved to %s", checkpoint_path)

    pretrain_dir = output_dir / "pretrain"
    pretrain_dir.mkdir(parents=True, exist_ok=True)
    pretrain_path = pretrain_dir / "dinov3_pretrain.pth"
    save_pretrain_checkpoint(
        pretrain_path,
        build_pretrain_checkpoint(
            bundle,
            cfg,
            iteration=total_iters,
            optimizer=optimizer,
            scaler=scaler,
            extra={
                "dino_loss": dino_loss.state_dict(),
                "ibot_loss": ibot_loss.state_dict(),
                "gram_updates_done": gram_updates_done,
            },
        ),
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
    parser.add_argument(
        "--schedule-steps",
        type=int,
        default=None,
        help=(
            "Use this many iterations for LR/WD/teacher/Gram schedules while running --steps iterations. "
            "Useful for short probes that should preserve the full-run schedule."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Override per-GPU batch size")
    parser.add_argument("--num-workers", type=int, default=None, help="Override data loader workers")
    parser.add_argument("--seed", type=int, default=0, help="Global random seed for SSL and post-train")
    parser.add_argument(
        "--resume-pretrain",
        type=Path,
        default=None,
        help="Resume SSL pretraining from a checkpoint_*.pth or dinov3_pretrain.pth checkpoint.",
    )
    # Post-train (curve) stage
    parser.add_argument(
        "--post-train-steps", type=int, default=None, help="Run curve post-training for N steps (0 to skip)"
    )
    parser.add_argument("--post-train-batch-size", type=int, default=None, help="Batch size for post-training")
    parser.add_argument("--post-train-lr-head", type=float, default=None)
    parser.add_argument("--post-train-lr-lora", type=float, default=None)
    parser.add_argument("--post-train-wd-head", type=float, default=None)
    parser.add_argument("--post-train-wd-lora", type=float, default=None)
    parser.add_argument(
        "--post-train-lr-warmup",
        type=int,
        default=None,
        help="Warmup steps for post-train LR schedule",
    )
    parser.add_argument(
        "--post-train-min-lr-mult",
        type=float,
        default=None,
        help="Final LR multiplier for post-train cosine decay",
    )
    parser.add_argument(
        "--post-train-ema-decay",
        type=float,
        default=None,
        help="EMA decay",
    )
    parser.add_argument("--post-train-sigma", type=float, default=None)
    parser.add_argument(
        "--post-train-bg-weight",
        type=float,
        default=None,
        help="CE weight multiplier for background samples (LossCfg.bg_weight; default 5.0).",
    )
    parser.add_argument("--post-train-lambda-curve", type=float, default=None)
    parser.add_argument("--post-train-lambda-curv", type=float, default=None)
    parser.add_argument("--post-train-lora-blocks", type=int, default=None)
    parser.add_argument("--post-train-lora-r", type=int, default=None)
    parser.add_argument("--post-train-lora-alpha", type=int, default=None)
    parser.add_argument("--post-train-lora-dropout", type=float, default=None)
    parser.add_argument("--post-train-lora-use-mlp", action="store_true")
    parser.add_argument("--post-train-unfreeze-dwconv", action="store_true")
    parser.add_argument(
        "--post-train-backbone-norm-mode",
        choices=["all_norms", "final_only", "none"],
        default=None,
        help="Which backbone normalization layers remain trainable during post-train "
        "(all_norms=every norm, final_only=top-level backbone.norm, none=frozen).",
    )
    parser.add_argument("--post-train-curve-head-mid", type=int, default=None)
    parser.add_argument("--post-train-feature-layers", type=int, default=None)
    parser.add_argument("--post-train-input-aa-strength", type=float, default=None)
    parser.add_argument("--post-train-aug-p", type=float, default=None)
    parser.add_argument("--post-train-aug-types", nargs="+", default=None)
    parser.add_argument("--post-train-aug-severity", default=None)
    parser.add_argument(
        "--post-train-init-curve",
        type=Path,
        default=None,
        help="Initialize the post-train curve decoder from a compatible fused_curve checkpoint.",
    )
    parser.add_argument(
        "--post-train-method",
        choices=["sam", "adamw"],
        default=None,
        help="Override post-train optimizer method",
    )
    parser.add_argument(
        "--post-train-split-mode",
        choices=["auto", "all"],
        default=None,
        help="Post-train data handling: auto=use the explicit splits.csv assignment (fails "
        "closed when split metadata is missing); all=train on every labeled+background sample "
        "with no held-out split (deployment mode; val metrics become monitor-only train-set "
        "metrics).",
    )
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
    effective_seed = int(args.seed)
    cfg.setdefault("train", {})["seed"] = effective_seed
    fix_random_seeds(effective_seed)
    post_cfg = cfg.get("post_train", {})
    if args.batch_size is not None:
        cfg.setdefault("train", {})["batch_size_per_gpu"] = args.batch_size
    if args.num_workers is not None:
        cfg.setdefault("train", {})["num_workers"] = args.num_workers
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
        train(
            cfg,
            steps_override=args.steps,
            schedule_steps_override=args.schedule_steps,
            output_dir_override=output_dir,
            seed_override=effective_seed,
            resume_pretrain=args.resume_pretrain,
        )
    if post_steps and post_steps > 0:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for curve post-training.")
        device = torch.device("cuda")
        post_dir = output_dir / "post_train"
        setup_file_logging(post_dir / "post_train.log")
        backbone = None
        pretrain_path = output_dir / "pretrain" / "dinov3_pretrain.pth"
        ckpt_path = args.pretrained_backbone if (args.post_train_only and args.pretrained_backbone) else pretrain_path
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("student", ckpt.get("model", ckpt))
        # torch.compile saves params under a "_orig_mod." prefix; the post-train backbone is
        # built uncompiled, so strip it or every key silently fails to match (load_state_dict
        # is strict=False) and the backbone stays at random init.
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
        # Primary guard: assert the checkpoint's own recorded arch matches the build arch.
        # (Catches same-family size mismatches like convnext_small->convnext_tiny, whose keys
        # are a strict subset and would otherwise load 180/180 with unexpected!=0 and pass.)
        ckpt_cfg = ckpt.get("config") if isinstance(ckpt, dict) else None
        if isinstance(ckpt_cfg, dict):
            ck_arch = str(get_cfg(ckpt_cfg, ("student", "arch"), "") or "")
            ck_ps = get_cfg(ckpt_cfg, ("student", "patch_size"), None)
            bd_arch = str(get_cfg(cfg, ("student", "arch"), "") or "")
            bd_ps = get_cfg(cfg, ("student", "patch_size"), None)
            if ck_arch and bd_arch and ck_arch != bd_arch:
                raise RuntimeError(
                    f"post-train arch mismatch: checkpoint {ckpt_path} is '{ck_arch}' but build config is "
                    f"'{bd_arch}'. Pass --config with the matching arch; refusing to load a wrong-arch backbone."
                )
            if ck_ps is not None and bd_ps is not None and int(ck_ps) != int(bd_ps):
                raise RuntimeError(
                    f"post-train patch_size mismatch: checkpoint={ck_ps} vs build={bd_ps} ({ckpt_path})."
                )
        backbone = _build_backbone_from_cfg(cfg, torch.device("cpu"))
        missing, unexpected = backbone.load_state_dict(state, strict=False)
        n_keys = len(backbone.state_dict())
        n_loaded = n_keys - len(missing)
        # Backstop (covers checkpoints with no embedded config): require most keys to load AND
        # near-zero unexpected keys. A subset-arch load (small->tiny) shows n_loaded=100% but
        # unexpected>>0, so the unexpected check is essential, not redundant.
        if n_loaded < 0.5 * n_keys or len(unexpected) > 0.05 * n_keys:
            raise RuntimeError(
                f"post-train backbone load matched only {n_loaded}/{n_keys} keys from {ckpt_path} "
                f"(missing={len(missing)}, unexpected={len(unexpected)}) — arch/prefix mismatch; refusing "
                "to post-train on a wrong or random-init backbone."
            )
        logger.info(
            "loaded post-train backbone from %s: %d/%d keys (missing=%d, unexpected=%d)",
            ckpt_path, n_loaded, n_keys, len(missing), len(unexpected),
        )
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
        post_lr_warmup = (
            args.post_train_lr_warmup
            if args.post_train_lr_warmup is not None
            else post_cfg.get("lr_warmup", 50)
        )
        post_min_lr_mult = (
            args.post_train_min_lr_mult
            if args.post_train_min_lr_mult is not None
            else post_cfg.get("min_lr_mult", 0.1)
        )
        post_ema_decay = (
            args.post_train_ema_decay
            if args.post_train_ema_decay is not None
            else post_cfg.get("ema_decay", 0.0)
        )
        def post_arg(value, key: str, default):
            return value if value is not None else post_cfg.get(key, default)

        run_post_training(
            backbone=backbone,
            patch_size=int(get_cfg(cfg, ("student", "patch_size"), 14)),
            dataset_str=resolved_ds,
            seed=effective_seed,
            split_mode=str(args.post_train_split_mode or post_cfg.get("split_mode", "auto")),
            steps=int(post_steps),
            batch_size=int(args.post_train_batch_size or post_cfg.get("batch_size", 128)),
            num_workers=int(get_cfg(cfg, ("train", "num_workers"), 4)),
            lr_head=float(post_arg(args.post_train_lr_head, "lr_head", 1e-3)),
            wd_head=float(post_arg(args.post_train_wd_head, "wd_head", 5e-4)),
            lr_lora=float(post_arg(args.post_train_lr_lora, "lr_lora", 5e-4)),
            wd_lora=float(post_arg(args.post_train_wd_lora, "wd_lora", 0.0)),
            lr_warmup=int(post_lr_warmup),
            min_lr_mult=float(post_min_lr_mult),
            ema_decay=float(post_ema_decay),
            sigma=float(post_arg(args.post_train_sigma, "sigma", 1.5)),
            bg_weight=float(post_arg(args.post_train_bg_weight, "bg_weight", 5.0)),
            lambda_curve=float(post_arg(args.post_train_lambda_curve, "lambda_curve", 1.0)),
            lambda_curv=float(post_arg(args.post_train_lambda_curv, "lambda_curv", 0.05)),
            eps_none=float(post_cfg.get("eps_none", 0.02)),
            curv_delta=float(post_cfg.get("curv_delta", 1.0)),
            lora_blocks=int(post_arg(args.post_train_lora_blocks, "lora_blocks", 3)),
            lora_r=int(post_arg(args.post_train_lora_r, "lora_r", 8)),
            lora_alpha=int(post_arg(args.post_train_lora_alpha, "lora_alpha", 16)),
            lora_dropout=float(post_arg(args.post_train_lora_dropout, "lora_dropout", 0.05)),
            lora_use_mlp=bool(args.post_train_lora_use_mlp or post_cfg.get("lora_use_mlp", False)),
            unfreeze_dwconv=bool(args.post_train_unfreeze_dwconv or post_cfg.get("unfreeze_dwconv", False)),
            norm_mode=str(args.post_train_backbone_norm_mode or post_cfg.get("backbone_norm_mode", "all_norms")),
            curve_head_mid=int(post_arg(args.post_train_curve_head_mid, "curve_head_mid", 128)),
            feature_layers=int(post_arg(args.post_train_feature_layers, "feature_layers", 1)),
            input_aa_strength=float(post_arg(args.post_train_input_aa_strength, "input_aa_strength", 0.0)),
            train_aug_p=float(post_arg(args.post_train_aug_p, "train_aug_p", 0.0)),
            train_aug_types=(
                args.post_train_aug_types if args.post_train_aug_types is not None else post_cfg.get("train_aug_types", None)
            ),
            train_aug_severity=str(args.post_train_aug_severity or post_cfg.get("train_aug_severity", "medium")),
            init_curve_path=args.post_train_init_curve,
            method=str(args.post_train_method or post_cfg.get("method", "sam")),
            sam_rho=float(sam_rho),
            log_every=int(get_cfg(cfg, ("train", "log_every"), 10)),
            val_every=int(post_cfg.get("val_every", 1)),
            device=device,
            output_path=post_out,
            best_path=best_out,
        )
    if args.post_train_only and args.steps:
        logger.warning("Ignoring --steps when --post-train-only is set")


if __name__ == "__main__":
    main()
