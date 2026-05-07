"""Image-perturbation table on v2 (5053 items).

Rows  = (model, image_setting) — 3 image settings per model:
            Normal, No Image, Image Corruption
Cols  = 3 prompt settings: Direct, Reasoning, Rule-Given
Cell  = overall accuracy (%) across all 5053 items.

Output:
    tables/results_image_perturbation.tex
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
from _utils import composite_key, extract_ans

MODELS = [
    ('llava',     'LLaVA-1.6-7B'),
    ('gemma3',    'Gemma3-12B'),
    ('internvl3', 'InternVL3-8B'),
    ('qwen3vl',   'Qwen3-VL-8B'),
]
# image_setting → key prefix in the result filename.
# (None means the "normal" image baseline; no prefix.)
IMAGE_SETTINGS = [
    (None,               'Normal'),
    ('no_image',         'No Image'),
    ('image_corruption', 'Image Corruption'),
]
PROMPT_SETTINGS = [
    ('direct',     'Direct'),
    ('reasoning',  'Reasoning'),
    ('rule_given', 'Rule-Given'),
]


def file_for(model: str, img_pref, prompt_key: str) -> Path:
    """Map (model, image_setting, prompt_setting) → result-file path."""
    if img_pref is None:
        # normal image: <model>_<prompt>_v2_results.json
        stem = f'{model}_{prompt_key}'
    else:
        # perturbed image:
        #   direct      → <model>_<img_pref>_v2_results.json
        #   reasoning   → <model>_<img_pref>_reasoning_v2_results.json
        #   rule_given  → <model>_<img_pref>_rule_given_v2_results.json
        stem = (f'{model}_{img_pref}' if prompt_key == 'direct'
                else f'{model}_{img_pref}_{prompt_key}')
    return RESULT_DIR / f'{stem}_v2_results.json'


def overall_acc(model: str, img_pref, prompt_key: str, valid_keys) -> float:
    p = file_for(model, img_pref, prompt_key)
    if not p.exists():
        return np.nan
    rows = json.load(open(p))
    uniq = {composite_key(r): r for r in rows}
    correct = total = 0
    for k, r in uniq.items():
        if k not in valid_keys or r.get('gt') is None:
            continue
        total += 1
        if extract_ans(r.get('pred', '')) == r.get('gt'):
            correct += 1
    return (correct / total * 100) if total else np.nan


def main():
    valid_keys = {composite_key(x) for x in json.load(open(BENCHMARK))}

    # acc[model_idx][image_idx][prompt_idx] = float
    n_m, n_i, n_p = len(MODELS), len(IMAGE_SETTINGS), len(PROMPT_SETTINGS)
    acc = np.full((n_m, n_i, n_p), np.nan)
    for mi, (m_key, _) in enumerate(MODELS):
        for ii, (img_pref, _) in enumerate(IMAGE_SETTINGS):
            for pi, (prompt_key, _) in enumerate(PROMPT_SETTINGS):
                acc[mi, ii, pi] = overall_acc(m_key, img_pref, prompt_key, valid_keys)

    # Console
    print('\n=== Overall accuracy (%) — v2 (5053 items) ===')
    print(f'  {"":<26s} ' + '  '.join(f'{lbl:>11s}' for _, lbl in PROMPT_SETTINGS))
    for mi, (_, m_label) in enumerate(MODELS):
        for ii, (_, img_label) in enumerate(IMAGE_SETTINGS):
            row = '  '.join(
                (f'{acc[mi, ii, pi]:>11.2f}' if not np.isnan(acc[mi, ii, pi])
                 else f'{"--":>11s}')
                for pi in range(n_p)
            )
            print(f'  {m_label:<14s} {img_label:<11s} {row}')
        print()

    # LaTeX
    out = TABLE_DIR / 'results_image_perturbation.tex'
    col_spec = 'll' + 'c' * n_p
    body = []
    body.append(r'\begin{table}[t]')
    body.append(r'\centering')
    body.append(r'\small')
    body.append(r'\setlength{\tabcolsep}{5pt}')
    body.append(r'\caption{Robustness to image perturbation on '
                r'\textsc{CulturalDrive-Bench} v2 (5053 items). For each base '
                r'VLM we report overall accuracy (\%) under three input '
                r'conditions (\emph{Normal}: original image; \emph{No Image}: '
                r'image removed; \emph{Image Corruption}: heavy noise / blur '
                r'applied) crossed with three prompting strategies '
                r'(\emph{Direct}, \emph{Reasoning}, \emph{Rule-Given}). '
                r'Within each (model, prompt) column triplet we '
                r'\textbf{bold} the highest accuracy. Cells marked '
                r'``--\,\!" indicate the run is not yet available.}')
    body.append(r'\label{tab:image_perturbation}')
    body.append(r'\begin{tabular}{' + col_spec + r'}')
    body.append(r'\toprule')
    header = ['Model', 'Image Setting'] + [lbl for _, lbl in PROMPT_SETTINGS]
    body.append(' & '.join(header) + r' \\')
    body.append(r'\midrule')

    for mi, (_, m_label) in enumerate(MODELS):
        # for each (model, prompt-col) triplet, find the best image-setting row
        best_per_col = []
        for pi in range(n_p):
            col_vals = acc[mi, :, pi]
            best = int(np.nanargmax(col_vals)) if not np.all(np.isnan(col_vals)) else -1
            best_per_col.append(best)
        for ii, (_, img_label) in enumerate(IMAGE_SETTINGS):
            cells = []
            for pi in range(n_p):
                v = acc[mi, ii, pi]
                if np.isnan(v):
                    cells.append('--')
                else:
                    s = f'{v:.2f}'
                    cells.append(r'\textbf{' + s + '}' if ii == best_per_col[pi] else s)
            model_cell = (r'\multirow{3}{*}{' + m_label + '}') if ii == 0 else ''
            body.append(f'{model_cell} & {img_label} & ' + ' & '.join(cells) + r' \\')
        if mi < n_m - 1:
            body.append(r'\midrule')

    body.append(r'\bottomrule')
    body.append(r'\end{tabular}')
    body.append(r'\end{table}')
    out.write_text('\n'.join(body) + '\n')
    print(f'\nLaTeX table → {out}')


if __name__ == '__main__':
    main()
