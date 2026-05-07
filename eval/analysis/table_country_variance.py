"""Per-country accuracy std for the MAIN table.

Rows = the 3 main-table settings (direct / reasoning / rule_given).
Cols = same model list as the main table.
Cell = std (across 6 countries) of overall accuracy (%) — one accuracy per
country, computed across all 4 task categories.

Outputs:
  - prints console table
  - writes LaTeX snippet to tables/results_country_std.tex
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

_ANALYSIS_DIR = Path(__file__).resolve().parent
_EVAL_DIR     = _ANALYSIS_DIR.parent
_REPO_ROOT    = _EVAL_DIR.parent
RESULT_DIR    = _EVAL_DIR / 'results'
TABLE_DIR     = _ANALYSIS_DIR / 'tables'
TABLE_DIR.mkdir(exist_ok=True)
BENCHMARK     = _REPO_ROOT / 'culturebenchmark_eval_v2.json'

sys.path.insert(0, str(_EVAL_DIR))
from _utils import (CATEGORIES, COUNTRIES, SETTINGS, SETTING_LABELS,
                    composite_key, extract_ans)

# Mirror the model order used in make_tables.py — but on the v2 (5053-item)
# benchmark. GPT-5.4 / Qwen3-VL-235B are excluded because they have not been
# re-run on v2 main-table settings yet.
MODELS = ['llava', 'gemma3', 'internvl3', 'internvl35',
          'qwen25vl', 'qwen3vl', 'llama32',
          'qwen25vl_sdft_v2', 'internvl3_sdft_v2']
MODEL_LABELS = {
    'llava':             'LLaVA-1.6-7B',
    'gemma3':            'Gemma3-12B',
    'internvl3':         'InternVL3-8B',
    'internvl35':        'InternVL3.5-8B',
    'qwen25vl':          'Qwen2.5-VL-7B',
    'qwen3vl':           'Qwen3-VL-8B',
    'llama32':           'Llama-3.2-11B-V',
    'qwen25vl_sdft_v2':  r'DriveOPD$^{\dagger}$',
    'internvl3_sdft_v2': r'DriveOPD$^{\ddagger}$',
}


def load_bench_country():
    return {composite_key(x): x['country']
            for x in json.load(open(BENCHMARK))}


def per_country_acc(model: str, setting: str, bench_country):
    p = RESULT_DIR / f'{model}_{setting}_v2_results.json'
    if not p.exists():
        return None
    uniq = {composite_key(r): r for r in json.load(open(p))}
    correct = {c: 0 for c in COUNTRIES}
    total   = {c: 0 for c in COUNTRIES}
    for k, r in uniq.items():
        c = bench_country.get(k)
        if c not in correct or r.get('gt') is None:
            continue
        total[c] += 1
        if extract_ans(r.get('pred', '')) == r.get('gt'):
            correct[c] += 1
    return np.array([(correct[c] / total[c] * 100 if total[c] else np.nan)
                     for c in COUNTRIES])


def main():
    bench_country = load_bench_country()

    n_set, n_mod = len(SETTINGS), len(MODELS)
    perc = [[None] * n_mod for _ in range(n_set)]
    std_tab  = np.full((n_set, n_mod), np.nan)
    mean_tab = np.full_like(std_tab, np.nan)

    for si, s in enumerate(SETTINGS):
        for mi, m in enumerate(MODELS):
            accs = per_country_acc(m, s, bench_country)
            if accs is None:
                continue
            perc[si][mi]    = accs
            std_tab[si, mi]  = np.nanstd(accs, ddof=0)
            mean_tab[si, mi] = np.nanmean(accs)

    # Console: per-country accuracy per setting
    print('\n=== Per-country accuracy (%) ===')
    for si, s in enumerate(SETTINGS):
        print(f'\n[{SETTING_LABELS[s]}]')
        head = f'  {"":<26s} ' + '  '.join(f'{c.upper():>6s}' for c in COUNTRIES) \
               + f'  {"mean":>6s}  {"std":>6s}'
        print(head)
        for mi, m in enumerate(MODELS):
            accs = perc[si][mi]
            if accs is None:
                print(f'  {MODEL_LABELS[m]:<26s} (no results)')
                continue
            cells = '  '.join(f'{v:>6.2f}' for v in accs)
            print(f'  {MODEL_LABELS[m]:<26s} {cells}  '
                  f'{mean_tab[si, mi]:>6.2f}  {std_tab[si, mi]:>6.2f}')

    print('\n=== Std summary (rows=setting, cols=model) ===')
    print(f'  {"":<14s} ' + '  '.join(f'{MODEL_LABELS[m]:>26s}' for m in MODELS))
    for si, s in enumerate(SETTINGS):
        row = '  '.join(f'{std_tab[si, mi]:>26.2f}' for mi in range(n_mod))
        print(f'  {SETTING_LABELS[s]:<14s} {row}')

    # LaTeX table — std across 6 countries.
    out = TABLE_DIR / 'results_country_std.tex'
    col_spec = 'l' + 'c' * n_mod
    body = []
    body.append(r'\begin{table}[t]')
    body.append(r'\centering')
    body.append(r'\small')
    body.append(r'\setlength{\tabcolsep}{4pt}')
    body.append(r'\caption{Cross-country standard deviation of overall accuracy '
                r'(in \%) on \textsc{CulturalDrive-Bench} v2 (5053 items, 6 countries). '
                r'For each (setting, model) pair we compute one accuracy per country '
                r'(across all four task categories) and report the standard deviation '
                r'over the six values. Lower values indicate more uniform performance '
                r'across regions; per row, the model with the smallest std is '
                r'\textbf{bold}. DriveOPD$^{\dagger}$ and DriveOPD$^{\ddagger}$ are our '
                r'SDFT checkpoints based on Qwen2.5-VL-7B and InternVL3-8B '
                r'respectively.}')
    body.append(r'\label{tab:country_std}')
    body.append(r'\begin{tabular}{' + col_spec + r'}')
    body.append(r'\toprule')
    body.append('Setting & ' + ' & '.join(MODEL_LABELS[m] for m in MODELS) + r' \\')
    body.append(r'\midrule')
    for si, s in enumerate(SETTINGS):
        row_vals = std_tab[si]
        best = int(np.nanargmin(row_vals)) if not np.all(np.isnan(row_vals)) else -1
        cells = []
        for mi in range(n_mod):
            v = std_tab[si, mi]
            if np.isnan(v):
                cells.append('--')
            else:
                cells.append(r'\textbf{' + f'{v:.2f}' + '}' if mi == best
                             else f'{v:.2f}')
        body.append(f'{SETTING_LABELS[s]} & ' + ' & '.join(cells) + r' \\')
    body.append(r'\bottomrule')
    body.append(r'\end{tabular}')
    body.append(r'\end{table}')
    out.write_text('\n'.join(body) + '\n')
    print(f'\nLaTeX table → {out}')


if __name__ == '__main__':
    main()
