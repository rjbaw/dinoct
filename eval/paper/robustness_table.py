#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

STATIC_ROWS: list[tuple[str, dict[str, tuple[float, float]]]] = [
    (
        'GF',
        {
            'clean': (27.443, 20.009),
            'stripe_severe': (30.346, 19.779),
            'ghost_severe': (30.252, 21.951),
            'dropout_severe': (43.024, 19.954),
            'real_hard': (40.749, 12.653),
        },
    ),
    (
        'GRAD-ENG',
        {
            'clean': (13.050, 20.026),
            'stripe_severe': (17.043, 20.138),
            'ghost_severe': (18.768, 24.158),
            'dropout_severe': (16.427, 15.885),
            'real_hard': (15.386, 20.375),
        },
    ),
    (
        'TUNED-SOBEL-DC',
        {
            'clean': (5.635, 9.212),
            'stripe_severe': (40.074, 40.593),
            'ghost_severe': (18.028, 39.901),
            'dropout_severe': (14.550, 13.302),
            'real_hard': (125.342, 69.007),
        },
    ),
]

LEARNED_ROWS = [
    ('UNet', 'unet'),
    ('FCBR', 'fcbr'),
    ('DINOCT', 'dinoct'),
]


def _find_repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / 'pyproject.toml').exists() and (candidate / 'dinoct').is_dir():
            return candidate
    raise RuntimeError('Could not locate repo root from script path.')


REPO_ROOT = _find_repo_root()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Render the paper robustness table from frozen and verified summaries.')
    parser.add_argument(
        '--robustness-root',
        type=Path,
        default=REPO_ROOT / 'outputs' / 'paper_results' / 'robustness',
    )
    parser.add_argument(
        '--output-csv',
        type=Path,
        default=REPO_ROOT / 'outputs' / 'paper_results' / 'robustness' / 'robustness_table.csv',
    )
    parser.add_argument(
        '--output-tex',
        type=Path,
        default=REPO_ROOT / 'outputs' / 'paper_results' / 'robustness' / 'robustness_table.tex',
    )
    parser.add_argument(
        '--output-table-csv',
        type=Path,
        default=REPO_ROOT / 'outputs' / 'paper_tables' / 'robustness_explicit.csv',
    )
    parser.add_argument(
        '--output-table-tex',
        type=Path,
        default=REPO_ROOT / 'outputs' / 'paper_tables' / 'robustness_table.tex',
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_learned_row(root: Path, slug: str) -> dict[str, tuple[float, float]]:
    suite = {(r['corruption'], r['severity']): r for r in csv.DictReader((root / slug / 'robustness_suite_summary.csv').open())}
    hard = _read_json(root / slug / 'real_hard' / 'summary.json')
    hard_mean = float(hard['table_metrics_per_recording_mean']['mae_px'])
    hard_std = float(hard['table_metrics_per_recording_std']['mae_px'])
    return {
        'clean': (float(suite[('clean', 'medium')]['mae_px']), float(suite[('clean', 'medium')]['mae_px_std'])),
        'stripe_severe': (float(suite[('stripe', 'severe')]['mae_px']), float(suite[('stripe', 'severe')]['mae_px_std'])),
        'ghost_severe': (float(suite[('ghost', 'severe')]['mae_px']), float(suite[('ghost', 'severe')]['mae_px_std'])),
        'dropout_severe': (float(suite[('dropout', 'severe')]['mae_px']), float(suite[('dropout', 'severe')]['mae_px_std'])),
        'real_hard': (hard_mean, hard_std),
    }


def _fmt_pm(value: tuple[float, float]) -> str:
    mean, std = value
    return f'{mean:.3f} $\\pm$ {std:.3f}'


def _render_tex(rows: list[tuple[str, dict[str, tuple[float, float]]]]) -> str:
    lines = [
        r'\begin{table*}[tb]',
        r'\centering',
        r'\caption{Robustness performance MAE (px) (mean $\pm$ std) \(\downarrow\)}',
        r'\begin{tabularx}{\textwidth}{@{}Xccccc@{}}',
        r'\toprule',
        r' & Clean & Stripe-severe & Ghost-severe & Dropout-severe & Real difficult \\',
        r'\midrule',
    ]
    for label, metrics in rows:
        lines.append(
            f"{label} & "
            f"{_fmt_pm(metrics['clean'])} & "
            f"{_fmt_pm(metrics['stripe_severe'])} & "
            f"{_fmt_pm(metrics['ghost_severe'])} & "
            f"{_fmt_pm(metrics['dropout_severe'])} & "
            f"{_fmt_pm(metrics['real_hard'])} \\\\"
        )
    lines.extend([
        r'\bottomrule',
        r'\end{tabularx}',
        r'\label{tab:robustacc}',
        r'\end{table*}',
    ])
    return '\n'.join(lines) + '\n'


def main() -> None:
    args = parse_args()
    rows: list[tuple[str, dict[str, tuple[float, float]]]] = list(STATIC_ROWS)
    for label, slug in LEARNED_ROWS:
        rows.append((label, _load_learned_row(args.robustness_root, slug)))

    fieldnames = [
        'method',
        'clean_mean', 'clean_std',
        'stripe_severe_mean', 'stripe_severe_std',
        'ghost_severe_mean', 'ghost_severe_std',
        'dropout_severe_mean', 'dropout_severe_std',
        'real_hard_mean', 'real_hard_std',
    ]
    csv_rows = []
    for label, metrics in rows:
        csv_rows.append({
            'method': label,
            'clean_mean': metrics['clean'][0],
            'clean_std': metrics['clean'][1],
            'stripe_severe_mean': metrics['stripe_severe'][0],
            'stripe_severe_std': metrics['stripe_severe'][1],
            'ghost_severe_mean': metrics['ghost_severe'][0],
            'ghost_severe_std': metrics['ghost_severe'][1],
            'dropout_severe_mean': metrics['dropout_severe'][0],
            'dropout_severe_std': metrics['dropout_severe'][1],
            'real_hard_mean': metrics['real_hard'][0],
            'real_hard_std': metrics['real_hard'][1],
        })

    tex = _render_tex(rows)
    for path in [args.output_csv, args.output_table_csv]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
    for path in [args.output_tex, args.output_table_tex]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(tex)

    print(f'[robustness-table] wrote {args.output_csv}')
    print(f'[robustness-table] wrote {args.output_tex}')
    print(f'[robustness-table] wrote {args.output_table_csv}')
    print(f'[robustness-table] wrote {args.output_table_tex}')


if __name__ == '__main__':
    main()
