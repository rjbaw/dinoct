#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
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

SEVERE_CORRUPTIONS = ('stripe', 'ghost', 'dropout')
FAILURE_THRESHOLDS_PX = (5.0, 10.0)
AGGREGATE_METRIC_KEYS = [
    'clean_test_mae',
    'clean_test_p95',
    'clean_test_bias',
    'clean_test_acc2',
    'clean_test_spike_pct',
    'stripe_severe_mae',
    'ghost_severe_mae',
    'dropout_severe_mae',
    'severe_mean_mae',
    'worst_severe_mae',
    'real_hard_mae',
    'real_hard_p95',
    'real_hard_bias',
    'real_hard_acc2',
    'real_hard_spike_pct',
    'real_hard_failure_rate_5px',
    'real_hard_failure_rate_10px',
    'real_hard_p95_failure_rate_10px',
    'real_hard_spike_failure_rate',
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
    parser.add_argument('--real-hard-dir', type=Path, default=REPO_ROOT / 'data' / 'oct' / 'eval' / 'hard')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--seeds', type=int, nargs='+', default=None, help='Optional explicit seed list for multi-seed aggregation.')
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


def resolve_seeds(args: argparse.Namespace) -> list[int]:
    if args.seeds:
        return [int(seed) for seed in args.seeds]
    return [int(args.seed)]


def variant_run_dir(*, output_root: Path, variant_key: str, seed: int, multi_seed: bool) -> Path:
    base_dir = output_root / variant_key
    if not multi_seed:
        return base_dir
    return base_dir / 'seed_{}'.format(seed)


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


def robustness_metrics_from_summary(path: Path) -> dict[str, float]:
    rows = read_json(path)
    assert isinstance(rows, list)
    severe_rows = [row for row in rows if str(row.get('severity', '')).lower() == 'severe']
    if not severe_rows:
        raise ValueError('No severe rows found in {}'.format(path))

    out: dict[str, float] = {}
    severe_values: list[float] = []
    for corruption in SEVERE_CORRUPTIONS:
        row = next((item for item in severe_rows if str(item.get('corruption', '')).lower() == corruption), None)
        if row is None:
            raise ValueError('Missing severe {} row in {}'.format(corruption, path))
        value = float(row['mae_px'])
        out['{}_severe_mae'.format(corruption)] = value
        severe_values.append(value)
    out['severe_mean_mae'] = sum(severe_values) / float(len(severe_values))
    out['worst_severe_mae'] = max(severe_values)
    return out


def failure_metrics_from_scan_csv(path: Path) -> dict[str, float]:
    rows = list(csv.DictReader(path.open('r', newline='')))
    if not rows:
        raise ValueError('No per-scan rows found in {}'.format(path))
    n = float(len(rows))
    return {
        'real_hard_failure_rate_5px': sum(float(row['mae_px']) > FAILURE_THRESHOLDS_PX[0] for row in rows) / n,
        'real_hard_failure_rate_10px': sum(float(row['mae_px']) > FAILURE_THRESHOLDS_PX[1] for row in rows) / n,
        'real_hard_p95_failure_rate_10px': sum(float(row['p95_px']) > FAILURE_THRESHOLDS_PX[1] for row in rows) / n,
        'real_hard_spike_failure_rate': sum(float(row['spike_rate']) > 0.0 for row in rows) / n,
    }


def seed_mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError('Cannot aggregate an empty metric list')
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.fmean(values)), float(statistics.stdev(values))


def aggregate_variant_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError('Cannot aggregate an empty variant row list')
    aggregated: dict[str, Any] = {
        'recipe_key': rows[0]['recipe_key'],
        'label': rows[0]['label'],
        'change': rows[0]['change'],
        'num_seeds': len(rows),
        'seeds': '|'.join(str(int(row['seed'])) for row in rows),
        'train_dirs': '|'.join(str(row['train_dir']) for row in rows),
    }
    for key in AGGREGATE_METRIC_KEYS:
        mean_value, std_value = seed_mean_std([float(row[key]) for row in rows])
        aggregated[key] = mean_value
        aggregated['{}_seed_std'.format(key)] = std_value
    return aggregated


def fmt_seed_aggregate(mean_value: float, std_value: float, *, scale: float = 1.0, places: int = 3, multi_seed: bool) -> str:
    value = mean_value * scale
    if not multi_seed:
        return '{:.{places}f}'.format(value, places=places)
    return '{:.{places}f} $\\pm$ {:.{places}f}'.format(value, std_value * scale, places=places)


def render_table(rows: list[dict[str, Any]]) -> str:
    multi_seed = any(int(row.get('num_seeds', 1)) > 1 for row in rows)
    caption = 'Single-component ablations from the fixed reference training recipe. '
    if multi_seed:
        caption += 'Values are reported as mean $\\pm$ std across seeds of the corresponding evaluation summary metric. '
    else:
        caption += 'Values are reported from a single training seed. '
    caption += 'FailureRate@10px is computed on the held-out real-artifact stress set from per-B-scan predictions.'
    lines = [
        '\\begin{table*}[t]',
        '\\centering',
        '\\caption{' + caption + '}',
        '\\footnotesize',
        '\\begin{tabularx}{\\textwidth}{@{}lXcccccc@{}}',
        '\\toprule',
        'Recipe & Change from Full & Clean MAE & Severe mean MAE & Worst severe MAE & Real difficult MAE & Real difficult Acc@2px & FailureRate@10px (\\%) \\\\',
        '\\midrule',
    ]
    for row in rows:
        lines.append('{} & {} & {} & {} & {} & {} & {} & {} \\\\'.format(
            row['label'],
            row['change'],
            fmt_seed_aggregate(float(row['clean_test_mae']), float(row['clean_test_mae_seed_std']), multi_seed=multi_seed),
            fmt_seed_aggregate(float(row['severe_mean_mae']), float(row['severe_mean_mae_seed_std']), multi_seed=multi_seed),
            fmt_seed_aggregate(float(row['worst_severe_mae']), float(row['worst_severe_mae_seed_std']), multi_seed=multi_seed),
            fmt_seed_aggregate(float(row['real_hard_mae']), float(row['real_hard_mae_seed_std']), multi_seed=multi_seed),
            fmt_seed_aggregate(float(row['real_hard_acc2']), float(row['real_hard_acc2_seed_std']), multi_seed=multi_seed),
            fmt_seed_aggregate(
                float(row['real_hard_failure_rate_10px']),
                float(row['real_hard_failure_rate_10px_seed_std']),
                scale=100.0,
                places=1,
                multi_seed=multi_seed,
            ),
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
    seeds = resolve_seeds(args)
    multi_seed = len(seeds) > 1
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    per_seed_rows: list[dict[str, Any]] = []
    aggregated_rows: list[dict[str, Any]] = []

    for variant in VARIANTS:
        variant_rows: list[dict[str, Any]] = []
        for seed in seeds:
            variant_dir = variant_run_dir(output_root=output_root, variant_key=variant['key'], seed=seed, multi_seed=multi_seed)
            train_dir = variant_dir / 'train'
            eval_dir = variant_dir / 'eval'
            clean_summary = eval_dir / 'clean_test' / 'summary.json'
            hard_summary = eval_dir / 'real_hard' / 'summary.json'
            hard_scan_csv = eval_dir / 'real_hard' / 'per_scan_metrics.csv'
            robustness_summary = eval_dir / 'robustness' / 'robustness_suite_summary.json'

            ready = clean_summary.exists() and hard_summary.exists() and hard_scan_csv.exists() and robustness_summary.exists()
            if not (args.resume and ready):
                train_cmd = [
                    sys.executable,
                    '-m',
                    'dinoct',
                    '--config', str(args.config),
                    '--output-dir', str(train_dir),
                    '--seed', str(seed),
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
                    '--eval-dir', str(args.real_hard_dir),
                    '--output-dir', str(eval_dir / 'real_hard'),
                    '--device', str(args.device),
                    '--batch-size', str(args.eval_batch_size),
                    '--num-workers', str(args.eval_num_workers),
                ]
                run(hard_cmd)

                robust_cmd = [
                    sys.executable,
                    str(REPO_ROOT / 'eval' / 'paper' / 'robustness.py'),
                    '--name', '{}_seed_{}'.format(variant['key'], seed),
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
            robust = robustness_metrics_from_summary(robustness_summary)
            failures = failure_metrics_from_scan_csv(hard_scan_csv)

            row = {
                'recipe_key': variant['key'],
                'seed': int(seed),
                'label': variant['label'],
                'change': variant['change'],
                'clean_test_mae': clean['mae'],
                'clean_test_p95': clean['p95'],
                'clean_test_bias': clean['bias'],
                'clean_test_acc2': clean['acc2'],
                'clean_test_spike_pct': clean['spike_pct'],
                'stripe_severe_mae': robust['stripe_severe_mae'],
                'ghost_severe_mae': robust['ghost_severe_mae'],
                'dropout_severe_mae': robust['dropout_severe_mae'],
                'severe_mean_mae': robust['severe_mean_mae'],
                'worst_severe_mae': robust['worst_severe_mae'],
                'real_hard_mae': hard['mae'],
                'real_hard_p95': hard['p95'],
                'real_hard_bias': hard['bias'],
                'real_hard_acc2': hard['acc2'],
                'real_hard_spike_pct': hard['spike_pct'],
                'real_hard_failure_rate_5px': failures['real_hard_failure_rate_5px'],
                'real_hard_failure_rate_10px': failures['real_hard_failure_rate_10px'],
                'real_hard_p95_failure_rate_10px': failures['real_hard_p95_failure_rate_10px'],
                'real_hard_spike_failure_rate': failures['real_hard_spike_failure_rate'],
                'train_dir': str(train_dir),
            }
            per_seed_rows.append(row)
            variant_rows.append(row)

        aggregated_rows.append(aggregate_variant_rows(variant_rows))

    per_seed_csv_path = output_root / 'final_component_ablation_per_seed.csv'
    with per_seed_csv_path.open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(per_seed_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_seed_rows)

    aggregated_csv_path = output_root / 'final_component_ablation.csv'
    with aggregated_csv_path.open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(aggregated_rows[0].keys()))
        writer.writeheader()
        writer.writerows(aggregated_rows)

    tex_path = output_root / 'final_component_ablation.tex'
    tex_path.write_text(render_table(aggregated_rows))

    print('[final-ablation] wrote {}'.format(per_seed_csv_path))
    print('[final-ablation] wrote {}'.format(aggregated_csv_path))
    print('[final-ablation] wrote {}'.format(tex_path))


if __name__ == '__main__':
    main()
