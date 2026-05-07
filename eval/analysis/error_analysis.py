"""Error-type analysis on a single (model, setting) by asking a SOTA VLM to
classify each wrong prediction into one of 6 error types.

Auto-classified with no API call:
    E5 — Refusal     (output contains "I cannot" / "I'm sorry" etc.)
    E6 — Format      (no A-D letter extractable AND not a refusal)

API-classified (GPT-5.4 / gpt-4o on each error's image + CoT):
    E1 — Visual misperception
    E2 — Geographic misclassification
    E3 — Cultural rule gap
    E4 — Reasoning error

Usage
-----
    OPENAI_API_KEY=sk-... \
    python error_analysis.py --model qwen3vl --setting reasoning \
        --analyzer gpt-4o --n_sample 100 --workers 8

Outputs
    eval/results/error_analysis_<model>_<setting>.json   # per-sample labels
    eval/plots/error_types_<model>_<setting>.png         # stacked bar by cat
"""
import argparse
import base64
import json
import mimetypes
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_ANALYSIS_DIR = Path(__file__).resolve().parent       # eval/analysis/
_EVAL_DIR     = _ANALYSIS_DIR.parent                  # eval/
_REPO_ROOT    = _EVAL_DIR.parent                      # repo root
RESULT_DIR    = _EVAL_DIR / 'results'
BENCH_PATH    = _REPO_ROOT / 'culturebenchmark_eval.json'
OUT_DIR       = _ANALYSIS_DIR / 'plots'
OUT_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(_EVAL_DIR))
from _utils import (CATEGORIES, composite_key, extract_ans, load_json,
                    save_json)

ERROR_TYPES = {
    'E1': 'Visual misperception',
    'E2': 'Geographic misclassification',
    'E3': 'Cultural rule gap',
    'E4': 'Reasoning error',
    'E5': 'Refusal',
    'E6': 'Format error',
}

REFUSAL_TOKENS = (
    "i cannot", "i can't", "i can not", "i'm sorry", "i am sorry",
    "i won't", "i refuse", "i'm not able", "i am not able", "i don't feel",
)

CLASSIFY_PROMPT = """You are auditing why a vision-language model answered a cross-cultural driving multiple-choice question incorrectly.

Country (ground truth, the rule jurisdiction this question concerns): {country}
Question: {q}
Options:
{opts}
Ground-truth answer: {gt}

Model's chain-of-thought (verbatim):
\"\"\"
{pred}
\"\"\"
Model's final letter: {ext}

Classify the SINGLE most-likely root cause of this error into ONE of:
- E1 (Visual misperception): the model misreads the image — wrong color/count/direction/sign text/object identity/lane.
- E2 (Geographic misclassification): the model inferred the wrong country/region in Step 1 (different from the ground-truth country above).
- E3 (Cultural rule gap): the model named the correct country but cited a wrong, fabricated, or irrelevant traffic rule for that country in Step 2.
- E4 (Reasoning error): visual observation and the cited rule are basically correct for the ground-truth country, but Step 3/4 produces the wrong answer (broken logic or picks a plausible-but-wrong distractor).

Reply with ONLY a JSON object: {{"category": "E1"|"E2"|"E3"|"E4", "rationale": "<one short sentence>"}}.
No prose outside the JSON."""


# ============================================================ helpers =====
def _data_uri(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    mime = mime or 'image/jpeg'
    b64 = base64.b64encode(open(path, 'rb').read()).decode('ascii')
    return f'data:{mime};base64,{b64}'


def is_refusal(pred: str) -> bool:
    s = str(pred or '').lower()
    return any(t in s for t in REFUSAL_TOKENS)


def auto_class(pred: str):
    """Return E5/E6/None (None = needs API)."""
    if is_refusal(pred):
        return 'E5'
    if extract_ans(pred) is None:
        return 'E6'
    return None


# ============================================================== OpenAI ====
_cli = None
def _client():
    global _cli
    if _cli is None:
        from openai import OpenAI
        _cli = OpenAI(api_key=os.environ['OPENAI_API_KEY'].strip())
    return _cli


def classify_via_api(model: str, item, image_paths, reasoning_effort='minimal',
                     max_retries=3, timeout=120):
    q       = item['question']
    opts    = "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(item['options']))
    gt      = item['gt']
    pred    = str(item.get('pred', ''))[:2000]  # cap CoT
    ext     = extract_ans(pred) or '?'
    country = item.get('country', '?')
    text    = CLASSIFY_PROMPT.format(country=country, q=q, opts=opts, gt=gt,
                                     pred=pred, ext=ext)

    content = [{'type': 'image_url', 'image_url': {'url': _data_uri(p)}}
               for p in image_paths]
    content.append({'type': 'text', 'text': text})
    msgs = [{'role': 'user', 'content': content}]

    backoff = 2.0
    last = None
    use_reasoning = reasoning_effort and reasoning_effort != 'none'
    for _ in range(max_retries):
        try:
            kwargs = dict(model=model, messages=msgs, timeout=timeout,
                          max_completion_tokens=2000)
            if use_reasoning:
                try:
                    r = _client().chat.completions.create(
                        reasoning_effort=reasoning_effort, **kwargs)
                except (TypeError, Exception) as _e:
                    # If model rejects reasoning_effort value, drop the param.
                    if 'reasoning_effort' in str(_e) and 'unsupported' in str(_e).lower():
                        r = _client().chat.completions.create(**kwargs)
                    else:
                        raise
            else:
                r = _client().chat.completions.create(**kwargs)
            txt = (r.choices[0].message.content or '').strip()
            m = re.search(r'\{.*\}', txt, re.DOTALL)
            if not m:
                last = f'no JSON in: {txt[:100]}'
                continue
            obj = json.loads(m.group(0))
            cat = obj.get('category', '').upper()
            if cat in {'E1', 'E2', 'E3', 'E4'}:
                return {'category': cat, 'rationale': obj.get('rationale', '')}
            last = f'bad category: {cat}'
        except Exception as e:
            last = str(e)
            time.sleep(backoff); backoff = min(backoff * 2, 20)
    return {'category': 'E?', 'rationale': f'classifier failed: {last}'}


# =============================================================== main =====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model',    required=True,
                    help='Result file: results/<model>_<setting>_results.json')
    ap.add_argument('--setting',  required=True)
    ap.add_argument('--analyzer', default='gpt-5.4',
                    help='OpenAI model used as classifier (gpt-4o, gpt-5.4, ...)')
    ap.add_argument('--reasoning_effort', default='low',
                    choices=['none','minimal','low','medium','high','xhigh'],
                    help='gpt-5* only (gpt-5.4 supports none/low/medium/high/xhigh)')
    ap.add_argument('--per_country',  type=int, default=60,
                    help='Sample up to this many error items per country.')
    ap.add_argument('--per_category', type=int, default=60,
                    help='Sample up to this many error items per question category.')
    ap.add_argument('--workers',  type=int, default=8)
    ap.add_argument('--seed',     type=int, default=42)
    ap.add_argument('--v2', action='store_true',
                    help='Use v2 benchmark + result file suffix.')
    ap.add_argument('--bench', default=None,
                    help='Override benchmark JSON path.')
    args = ap.parse_args()

    bench_path = args.bench or (
        str(_REPO_ROOT / 'culturebenchmark_eval_v2.json') if args.v2
        else BENCH_PATH)
    suffix = '_v2_results.json' if args.v2 else '_results.json'

    bench   = load_json(bench_path)
    bench_x = {composite_key(b): b for b in bench}

    results = load_json(RESULT_DIR / f'{args.model}_{args.setting}{suffix}')
    # Dedup by composite_key for fairness.
    uniq = {composite_key(r): r for r in results}
    wrong = [r for r in uniq.values()
             if r.get('gt') is not None and extract_ans(r.get('pred', '')) != r.get('gt')]

    print(f'[{args.model}/{args.setting}]  total errors: {len(wrong)}')

    # Auto-classify E5/E6 first.
    auto_labels = {}
    api_pool = []
    for r in wrong:
        cat = auto_class(r.get('pred', ''))
        k = composite_key(r)
        if cat is not None:
            auto_labels[k] = {'category': cat, 'rationale': '(auto)'}
        else:
            api_pool.append(r)

    print(f'  auto E5 (refusal):    {sum(1 for v in auto_labels.values() if v["category"]=="E5")}')
    print(f'  auto E6 (format):     {sum(1 for v in auto_labels.values() if v["category"]=="E6")}')
    print(f'  needs API (E1-E4):    {len(api_pool)}')

    # Sampling: up to N per country AND up to N per category; union (overlap OK).
    by_country  = defaultdict(list)
    by_category = defaultdict(list)
    for r in api_pool:
        info = bench_x.get(composite_key(r))
        if info is None:
            continue
        by_country[info['country']].append(r)
        by_category[info['question_category']].append(r)

    rng = random.Random(args.seed)
    picked_keys = set()
    pick = []
    def _take(bucket_dict, cap):
        for key in sorted(bucket_dict.keys()):
            pool = bucket_dict[key]
            n = min(cap, len(pool))
            for r in rng.sample(pool, n):
                k = composite_key(r)
                if k in picked_keys:
                    continue
                picked_keys.add(k)
                pick.append(r)
    _take(by_country,  args.per_country)
    _take(by_category, args.per_category)

    print(f'  sampled for API: {len(pick)} unique items '
          f'(union of {args.per_country}/country × {len(by_country)} '
          f'+ {args.per_category}/category × {len(by_category)})')

    # Call API in parallel.
    api_labels = {}
    def _job(r):
        info = bench_x[composite_key(r)]
        imgs = info['image_path'][:1]   # analyzer needs only the first image
        item = {**r, 'gt': r['gt'], 'country': info.get('country', '?')}
        out = classify_via_api(args.analyzer, item, imgs,
                               reasoning_effort=args.reasoning_effort)
        return composite_key(r), out, r

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_job, r) for r in pick]
        for i, fut in enumerate(as_completed(futs), 1):
            k, label, r = fut.result()
            api_labels[k] = label
            if i % 10 == 0:
                print(f'    progress {i}/{len(pick)}')

    # Compose per-sample annotation and save.
    rows = []
    def _row(r, label):
        info = bench_x.get(composite_key(r), {})
        return {
            'composite_key': list(composite_key(r)),
            'country':       info.get('country'),
            'category':      info.get('question_category'),
            'gt':            r.get('gt'),
            'pred':          r.get('pred'),
            'extracted':     extract_ans(r.get('pred', '')),
            'error_type':    label['category'],
            'rationale':     label.get('rationale', ''),
        }
    for r in wrong:
        k = composite_key(r)
        if k in auto_labels:
            rows.append(_row(r, auto_labels[k]))
    for r in pick:
        k = composite_key(r)
        if k in api_labels:
            rows.append(_row(r, api_labels[k]))

    out_suffix = '_v2' if args.v2 else ''
    out_path = RESULT_DIR / f'error_analysis_{args.model}_{args.setting}{out_suffix}.json'
    save_json({'model': args.model, 'setting': args.setting,
               'analyzer': args.analyzer,
               'n_total_wrong': len(wrong),
               'n_auto_E5': sum(1 for v in auto_labels.values() if v['category']=='E5'),
               'n_auto_E6': sum(1 for v in auto_labels.values() if v['category']=='E6'),
               'n_api_sampled': len(pick),
               'rows': rows}, str(out_path))
    print(f'\nsaved per-sample annotations → {out_path}')

    # Summary.
    api_cnt = Counter(v['category'] for v in api_labels.values())
    print('\n=== API-sampled E1-E4 distribution ===')
    for k in ('E1','E2','E3','E4','E?'):
        print(f'  {k} ({ERROR_TYPES.get(k, "?"):28s}): {api_cnt.get(k, 0):>3d} '
              f'({100*api_cnt.get(k,0)/max(1,len(pick)):.1f}%)')

    # Per-country and per-category breakdown of sampled E1-E4 labels.
    print('\n=== sampled label distribution by country ===')
    by_country_lbl = defaultdict(Counter)
    by_cat_lbl     = defaultdict(Counter)
    for r in pick:
        k = composite_key(r)
        if k not in api_labels: continue
        info = bench_x.get(k, {})
        by_country_lbl[info.get('country')][api_labels[k]['category']] += 1
        by_cat_lbl[info.get('question_category')][api_labels[k]['category']] += 1
    for c in sorted(by_country_lbl):
        cnts = by_country_lbl[c]
        tot = sum(cnts.values())
        print(f'  {c:>4s}  total={tot:>3d}  ' +
              '  '.join(f'{k}={cnts.get(k,0)}' for k in ('E1','E2','E3','E4')))
    print()
    for c in sorted(by_cat_lbl):
        cnts = by_cat_lbl[c]
        tot = sum(cnts.values())
        print(f'  {c:>11s}  total={tot:>3d}  ' +
              '  '.join(f'{k}={cnts.get(k,0)}' for k in ('E1','E2','E3','E4')))


if __name__ == '__main__':
    main()
