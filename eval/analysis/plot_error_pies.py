"""2 figures (qwen2.5 family, internvl3 family). Each is a 2x6 grid of pie
charts: rows = {base, SDFT}, cols = 6 countries.

Style: pastel slices with white gaps, label inside slice with (name, %).

Outputs
    plots/fig_error_pies_qwen25vl.{png,pdf}
    plots/fig_error_pies_internvl3.{png,pdf}
"""
import json, os, sys
from pathlib import Path
from collections import defaultdict, Counter

import matplotlib.pyplot as plt
import numpy as np

_ANALYSIS_DIR = Path(__file__).resolve().parent
_EVAL_DIR     = _ANALYSIS_DIR.parent
RESULT_DIR    = _EVAL_DIR / 'results'
OUT_DIR       = _ANALYSIS_DIR / 'plots'
OUT_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(_EVAL_DIR))
from _utils import composite_key

COUNTRIES = ['cn', 'us', 'uk', 'jp', 'sg', 'ind']
COUNTRY_LABELS = {'cn':'China','us':'United States','uk':'United Kingdom',
                  'jp':'Japan','sg':'Singapore','ind':'India'}

# E1-E4 only; E5/E6 are tiny (<4%) and not informative per country.
TYPES = ['E1', 'E2', 'E3', 'E4']
TYPE_LABEL = {
    'E1': 'Visual\nMisperception',
    'E2': 'Geographic\nMisclass.',
    'E3': 'Cultural\nRule Gap',
    'E4': 'Reasoning\nError',
}
COLORS = {
    'E1': '#F4C770',  # warm gold
    'E2': '#A8C8E1',  # sky blue
    'E3': '#E08A86',  # coral
    'E4': '#7DBFA8',  # sage green
}

PAIRS = [
    {  # Qwen2.5-VL family
        'name':  'qwen25vl',
        'base':  ('qwen25vl',          'Qwen2.5-VL-7B'),
        'sdft':  ('qwen25vl_sdft_v2',  r'DriveOPD$^{\dagger}$'),
    },
    {  # InternVL3 family
        'name':  'internvl3',
        'base':  ('internvl3',          'InternVL3-8B'),
        'sdft':  ('internvl3_sdft_v2',  r'DriveOPD$^{\ddagger}$'),
    },
]


def load_country_counts(model: str, setting: str = 'reasoning'):
    """Read per-sample annotations file, return dict[country] -> Counter(E1..E4)."""
    p = RESULT_DIR / f'error_analysis_{model}_{setting}_v2.json'
    obj = json.load(open(p))
    out = defaultdict(Counter)
    for r in obj['rows']:
        c = r.get('country')
        et = r.get('error_type')
        if et in TYPES and c:
            out[c][et] += 1
    return out


def _autopct(pct):
    return f'{pct:.0f}%' if pct >= 3 else ''


def render_grid(pair):
    base_key,  base_label  = pair['base']
    sdft_key,  sdft_label  = pair['sdft']
    base_cnt = load_country_counts(base_key)
    sdft_cnt = load_country_counts(sdft_key)

    fig, axes = plt.subplots(2, 6, figsize=(17, 5.6), dpi=160)

    for col, c in enumerate(COUNTRIES):
        for row, (cnt, model_label) in enumerate([
            (base_cnt, base_label), (sdft_cnt, sdft_label)
        ]):
            ax = axes[row, col]
            sizes  = [cnt[c].get(t, 0) for t in TYPES]
            colors = [COLORS[t] for t in TYPES]
            total  = sum(sizes)
            if total == 0:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center')
                ax.set_axis_off(); continue
            wedges, _, autotexts = ax.pie(
                sizes, colors=colors, startangle=90, radius=1.0,
                wedgeprops=dict(edgecolor='white', linewidth=2.5),
                autopct=_autopct, pctdistance=0.7,
                textprops=dict(fontsize=10),
            )
            for at, sz in zip(autotexts, sizes):
                at.set_color('#222'); at.set_weight('bold')
                if sz / total < 0.03:
                    at.set_text('')
            ax.set_xlim(-1.15, 1.15)
            ax.set_ylim(-1.15, 1.15)
            ax.set_aspect('equal')

            if row == 0:
                ax.set_title(COUNTRY_LABELS[c], fontsize=13, pad=4, weight='bold')
            if col == 0:
                ax.text(-0.10, 0.5, model_label, transform=ax.transAxes,
                        rotation=90, ha='center', va='center',
                        fontsize=12, weight='bold')

    legend_elems = [
        plt.matplotlib.patches.Patch(facecolor=COLORS[t], edgecolor='white',
                                     label=TYPE_LABEL[t].replace('\n', ' '))
        for t in TYPES
    ]
    fig.legend(handles=legend_elems, loc='upper center',
               bbox_to_anchor=(0.5, 1.02), ncol=4, fontsize=12,
               frameon=True, columnspacing=2.0,
               handlelength=1.6, handletextpad=0.6)
    fig.subplots_adjust(top=0.88, bottom=0.02, left=0.025, right=0.995,
                        wspace=-0.25, hspace=0.10)

    out_png = OUT_DIR / f"fig_error_pies_{pair['name']}.png"
    out_pdf = OUT_DIR / f"fig_error_pies_{pair['name']}.pdf"
    plt.savefig(out_png, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    print(f'saved: {out_png}\n       {out_pdf}')


def main():
    for pair in PAIRS:
        render_grid(pair)


if __name__ == '__main__':
    main()
