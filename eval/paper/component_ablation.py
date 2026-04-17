#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

VARIANTS = [
    {
        'key': 'full',
        'label': 'Full',
        'change': 'Reference recipe',
        'overrides': [],
    },
    {
        'key': 'no_ema',
        'label': '-EMA',
        'change': '$\\mathrm{EMA}=0$',
        'overrides': ['--post-train-ema-decay', '0.0'],
    },
    {
        'key': 'no_curvature',
        'label': '-Curvature',
        'change': '$\\lambda_{\\mathrm{smooth}}=0$',
        'overrides': ['--post-train-lambda-curv', '0.0'],
    },
    {
        'key': 'no_lora',
        'label': '-LoRA',
        'change': '$\\mathrm{LoRA}=0$',
        'overrides': ['--post-train-lora-blocks', '0'],
    },
    {
        'key': 'adamw',
        'label': 'AdamW',
        'change': 'SAM $\\rightarrow$ AdamW',
        'overrides': ['--post-train-method', 'adamw'],
    },
    {
        'key': 'with_lr_schedule',
        'label': '+LR schedule',
        'change': 'lr\\_warmup=10, min\\_lr\\_mult=0.95',
        'overrides': ['--post-train-lr-warmup', '10', '--post-train-min-lr-mult', '0.95'],
    },
]


def repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / 'pyproject.toml').exists() and (parent / 'dinoct').is_dir():
            return parent
    raise RuntimeError('Could not locate repo root')


REPO_ROOT = repo_root()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run final single-component ablations from current oct.yaml')
    parser.add_argument('--config', type=Path, default=REPO_ROOT / 'configs' / 'train' / 'oct.yaml')
    parser.add_argument('--pretrained-backbone', type=Path, default=REPO_ROOT / 'outputs' / 'pretrain' / 'dinov3_pretrain.pth')
    parser.add_argument('--output-root', type=Path, default=REPO_ROOT / 'outputs' / 'paper_results' / 'ablations' / 'final_component_ablation')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--train-num-workers', type=int, default=0)
    parser.add_argument('--eval-batch-size', type=int, default=32)
    parser.add_argument('--eval-num-workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any] | list[dict[str, Any]]:
    return json.loads(path.read_text())


def run(cmd: list[str]) -> None:
    print('[final-ablation] running: {}'.format(' '.join(cmd)))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def metrics_from_summary(path: Path) -> dict[str, float]:
    summary = read_json(path)
    assert isinstance(summary, dict)
    mean_block = summary['table_metrics_per_recording_mean']
    return {
        'mae': float(mean_block['mae_px']),
        'p95': float(mean_block['p95_px']),
        'bias': float(mean_block['bias_px']),
        'acc2': float(mean_block['acc_2px']),
        'spike_pct': 100.0 * float(mean_block['spike_rate']),
    }


def severe_mean_from_robustness(path: Path) -> float:
    rows = read_json(path)
    assert isinstance(rows, list)
    severe = [row for row in rows if str(row.get('severity', '')).lower() == 'severe']
    if not severe:
        raise ValueError('No severe rows found in {}'.format(path))
    return sum(float(row['mae_px']) for row in severe) / float(len(severe))


def fmt(value: float) -> str:
    return '{:.3f}'.format(value)


def render_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        '\\begin{table*}[t]',
        '\\centering',
        '\\caption{Single-component ablations from the fixed reference training recipe.}',
        '\\footnotesize',
        '\\begin{tabularx}{\\textwidth}{@{}lrXXX@{}}',
        '\\toprule',
        'Recipe & Change from Full & Clean Test MAE & Real difficult MAE & Synthetic artifact Mean MAE \\\\',
        '\\midrule',
    ]
    for row in rows:
        lines.append('{} & {} & {} & {} & {} \\\\'.format(
            row['label'],
            row['change'],
            fmt(row['clean_test_mae']),
            fmt(row['real_hard_mae']),
            fmt(row['severe_mean_mae']),
        ))
    lines.extend([
        '\\bottomrule',
        '\\end{tabularx}',
        '\\label{tab:component_ablation_main}',
        '\\end{table*}',
    ])
    return '\n'.join(lines) + '\n'


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    result_rows: list[dict[str, Any]] = []

    for variant in VARIANTS:
        variant_dir = output_root / variant['key']
        train_dir = variant_dir / 'train'
        eval_dir = variant_dir / 'eval'
        clean_summary = eval_dir / 'clean_test' / 'summary.json'
        hard_summary = eval_dir / 'real_hard' / 'summary.json'
        robustness_summary = eval_dir / 'robustness' / 'robustness_suite_summary.json'

        ready = clean_summary.exists() and hard_summary.exists() and robustness_summary.exists()
        if not (args.resume and ready):
            train_cmd = [
                sys.executable,
                '-m',
                'dinoct',
                '--config', str(args.config),
                '--output-dir', str(train_dir),
                '--seed', str(args.seed),
                '--post-train-only',
                '--pretrained-backbone', str(args.pretrained_backbone),
                '--post-train-batch-size', str(args.batch_size),
                '--num-workers', str(args.train_num_workers),
            ]
            train_cmd.extend(variant['overrides'])
            run(train_cmd)

            ckpt = train_dir / 'post_train' / 'fused_curve_best.pth'

            clean_cmd = [
                sys.executable,
                str(REPO_ROOT / 'eval' / 'evaluate_curve.py'),
                '--config', str(args.config),
                '--curve-ckpt', str(ckpt),
                '--split', 'test',
                '--output-dir', str(eval_dir / 'clean_test'),
                '--device', str(args.device),
                '--batch-size', str(args.eval_batch_size),
                '--num-workers', str(args.eval_num_workers),
            ]
            run(clean_cmd)

            hard_cmd = [
                sys.executable,
                str(REPO_ROOT / 'eval' / 'evaluate_curve.py'),
                '--config', str(args.config),
                '--curve-ckpt', str(ckpt),
                '--eval-dir', str(REPO_ROOT / 'data' / 'oct' / 'eval' / 'hard'),
                '--output-dir', str(eval_dir / 'real_hard'),
                '--device', str(args.device),
                '--batch-size', str(args.eval_batch_size),
                '--num-workers', str(args.eval_num_workers),
            ]
            run(hard_cmd)

            robust_cmd = [
                sys.executable,
                str(REPO_ROOT / 'eval' / 'paper' / 'robustness.py'),
                '--name', variant['key'],
                '--config', str(args.config),
                '--curve-ckpt', str(ckpt),
                '--split', 'test',
                '--corruptions', 'stripe', 'ghost', 'dropout',
                '--severities', 'severe',
                '--output-dir', str(eval_dir / 'robustness'),
                '--device', str(args.device),
                '--batch-size', str(args.eval_batch_size),
                '--num-workers', str(args.eval_num_workers),
            ]
            run(robust_cmd)

        clean = metrics_from_summary(clean_summary)
        hard = metrics_from_summary(hard_summary)
        severe_mean = severe_mean_from_robustness(robustness_summary)
        result_rows.append({
            'recipe_key': variant['key'],
            'label': variant['label'],
            'change': variant['change'],
            'clean_test_mae': clean['mae'],
            'clean_test_p95': clean['p95'],
            'clean_test_bias': clean['bias'],
            'clean_test_acc2': clean['acc2'],
            'clean_test_spike_pct': clean['spike_pct'],
            'real_hard_mae': hard['mae'],
            'real_hard_p95': hard['p95'],
            'real_hard_bias': hard['bias'],
            'real_hard_acc2': hard['acc2'],
            'real_hard_spike_pct': hard['spike_pct'],
            'severe_mean_mae': severe_mean,
            'train_dir': str(train_dir),
        })

    csv_path = output_root / 'final_component_ablation.csv'
    with csv_path.open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(result_rows[0].keys()))
        writer.writeheader()
        writer.writerows(result_rows)

    tex_path = output_root / 'final_component_ablation.tex'
    tex_path.write_text(render_table(result_rows))

    print('[final-ablation] wrote {}'.format(csv_path))
    print('[final-ablation] wrote {}'.format(tex_path))


if __name__ == '__main__':
    main()
