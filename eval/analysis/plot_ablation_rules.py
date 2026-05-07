"""Bar chart: rule-context ablation per task category.

1×4 subplots (Perception / Prediction / Planning / Region). In each subplot,
x-axis = 5 models; 3 bars per model = Rule-Given / Wrong-Rule / Full-Handbook.

Outputs
    plots/fig_ablation_rules.{png,pdf}
"""
import json, os, sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_ANALYSIS_DIR = Path(__file__).resolve().parent
_EVAL_DIR     = _ANALYSIS_DIR.parent
_REPO_ROOT    = _EVAL_DIR.parent
RESULT_DIR    = _EVAL_DIR / 'results'
BENCHMARK     = _REPO_ROOT / 'culturebenchmark_eval_v2.json'
OUT_DIR       = _ANALYSIS_DIR / 'plots'
OUT_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(_EVAL_DIR))
from _utils import (CATEGORIES, CATEGORY_LABELS, composite_key, extract_ans)

# Order: bases first (cool tones), Ours last.
MODELS = [
    ('qwen3vl',           'Qwen3-VL-8B'),
    ('internvl3',         'InternVL3-8B'),
    ('gemma3',            'Gemma3-12B'),
    ('qwen25vl_sdft_v2',  r'DriveOPD$^{\dagger}$'),
    ('internvl3_sdft_v2', r'DriveOPD$^{\ddagger}$'),
]
# Three rule conditions; each gets its own bar color.
SETTINGS = [
    ('rule_given',    'Rule-Given',    '#9DB7D6'),  # cool blue
    ('wrong_rule',    'Wrong-Rule',    '#E89B9B'),  # rose (worst case)
    ('full_handbook', 'Full-Handbook', '#9FD89F'),  # green
]

# Pretty short model labels for the x-axis (multi-line for readability).
SHORT_LABELS = {
    'qwen3vl':           'Qwen3-VL\n8B',
    'internvl3':         'InternVL3\n8B',
    'gemma3':            'Gemma3\n12B',
    'qwen25vl_sdft_v2':  r'DriveOPD$^{\dagger}$',
    'internvl3_sdft_v2': r'DriveOPD$^{\ddagger}$',
}


def load_bench_cats():
    """Return composite_key -> question_category."""
    return {composite_key(x): x['question_category']
            for x in json.load(open(BENCHMARK))}


def acc_by_category(model: str, setting: str, bench_cat):
    """Return dict[category] = accuracy (0-1) for this (model, setting)."""
    p = RESULT_DIR / f'{model}_{setting}_v2_results.json'
    if not p.exists():
        return None
    uniq = {composite_key(r): r for r in json.load(open(p))}
    correct = {c: 0 for c in CATEGORIES}
    total   = {c: 0 for c in CATEGORIES}
    for k, r in uniq.items():
        cat = bench_cat.get(k)
        if cat not in correct or r.get('gt') is None:
            continue
        total[cat] += 1
        if extract_ans(r.get('pred', '')) == r.get('gt'):
            correct[cat] += 1
    return {c: (correct[c] / total[c] if total[c] else np.nan) for c in CATEGORIES}


def main():
    bench_cat = load_bench_cats()

    # data[cat][model_idx][setting_idx] = accuracy %
    data = {c: np.full((len(MODELS), len(SETTINGS)), np.nan) for c in CATEGORIES}
    for mi, (m_key, _) in enumerate(MODELS):
        for si, (s_key, _, _) in enumerate(SETTINGS):
            d = acc_by_category(m_key, s_key, bench_cat)
            if d is None:
                continue
            for c in CATEGORIES:
                data[c][mi, si] = d[c] * 100

    n_cats = len(CATEGORIES)
    n_models = len(MODELS)
    n_set = len(SETTINGS)
    bar_w = 0.25
    group_centers = np.arange(n_models)
    offsets = (np.arange(n_set) - (n_set - 1) / 2) * bar_w

    fig, axes = plt.subplots(1, n_cats, figsize=(20, 4.5), dpi=160, sharey=True)

    legend_handles = None
    for ax_i, cat in enumerate(CATEGORIES):
        ax = axes[ax_i]
        for si, (_, s_label, color) in enumerate(SETTINGS):
            x = group_centers + offsets[si]
            h = data[cat][:, si]
            bars = ax.bar(x, h, bar_w, color=color,
                          edgecolor='black', linewidth=0.7,
                          label=s_label, zorder=3)
            if ax_i == 0 and legend_handles is None:
                pass
        if legend_handles is None:
            legend_handles = ax.get_legend_handles_labels()

        ax.set_xticks(group_centers)
        ax.set_xticklabels([SHORT_LABELS[k] for k, _ in MODELS], fontsize=9)
        ax.set_title(CATEGORY_LABELS[cat] if False else cat.capitalize(),
                     fontsize=13, pad=8)
        ax.set_ylim(45, 100)
        ax.yaxis.grid(True, linestyle='--', linewidth=0.6,
                      color='#bbbbbb', zorder=0)
        ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.spines['left'].set_color('#888')
        ax.spines['bottom'].set_color('#888')
        if ax_i == 0:
            ax.set_ylabel('Accuracy (%)', fontsize=12)

    # Single legend at top-center.
    handles, labels = legend_handles
    leg = fig.legend(handles, labels, loc='upper center',
                     bbox_to_anchor=(0.5, 1.02), ncol=n_set,
                     frameon=True, fontsize=11,
                     columnspacing=1.6, handlelength=2.0, handletextpad=0.6)
    leg.get_frame().set_edgecolor('#bbbbbb')
    leg.get_frame().set_linewidth(0.8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = OUT_DIR / 'fig_ablation_rules.png'
    out_pdf = OUT_DIR / 'fig_ablation_rules.pdf'
    plt.savefig(out_png, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    print(f'saved: {out_png}\n       {out_pdf}')

    # Sanity print.
    for cat in CATEGORIES:
        print(f'\n[{cat}]  ' + '  '.join(f'{l:>13s}' for _, l, _ in SETTINGS))
        for mi, (_, m_label) in enumerate(MODELS):
            row = data[cat][mi]
            print(f'  {m_label:<26s} ' +
                  '  '.join(f'{v:>13.2f}' if not np.isnan(v) else f'{"—":>13s}'
                            for v in row))


if __name__ == '__main__':
    main()
