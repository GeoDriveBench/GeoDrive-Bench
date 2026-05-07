"""Build a PER-COUNTRY mapping from each region template_topic → handbook
section IDs. The previous global mapping failed because S1-S15 cover
different topics in different countries' handbooks (e.g., S11 = "yellow box"
in CN but "lane-control signals" in US). S16-S20 (added in augmentation)
ARE consistent.

Pipeline:
  1. For each country, GPT-5.4 sees:
       - the 20 section titles of THAT country's handbook
       - the 30 region template_topics + a representative question per topic
     and outputs: per-topic list of section IDs (in THAT country) that
     contain the rule needed to answer the question.
  2. Output saved to `eval/region_per_country_mapping.json`.
  3. Then `apply_to_benchmark()` writes per-item rule_reference based on
     (country, topic) pair.

Usage:
    OPENAI_API_KEY=... python eval/region_per_country_mapping.py
"""
import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

REPO = Path(__file__).resolve().parents[2]   # repo root (../../..)
HBDIR = REPO / 'traffic_handbook'
COUNTRIES = {
    'cn': 'mainland China',
    'us': 'United States',
    'uk': 'United Kingdom',
    'jp': 'Japan',
    'sg': 'Singapore',
    'ind': 'India',
}


def load_handbook(country):
    sys.path.insert(0, str(REPO))
    import importlib, traffic_handbook
    importlib.reload(traffic_handbook)
    name = {'cn':'china','us':'us','uk':'uk','jp':'japan','sg':'singapore','ind':'india'}[country]
    return getattr(traffic_handbook, f'{name}_traffic_handbook')


def section_titles(handbook, max_sections=20):
    """Return list of (section_id, FULL_paragraph) parsed from handbook string.
    GPT-5.4 needs the full content (not just title) so it can recognize when a
    section covers multiple sub-topics — e.g. S16 covers mobile, helmet, BAC,
    horn, headlight all in one paragraph; the title alone hides those details."""
    out = []
    for s_idx in range(1, max_sections + 1):
        sid = f'S{s_idx}'
        m = re.search(rf'({sid}\s.+?)(?=\nS\d+\s|\Z)', handbook, re.DOTALL)
        if not m: continue
        para = m.group(1).strip()
        # cap length to avoid blowing past prompt budget; keep enough to match topics
        out.append((sid, para[:700]))
    return out


PROMPT = """You are mapping driving-benchmark question topics to relevant
sections of a country-specific traffic handbook. The handbook for {country_name}
has the following 20 sections (S1-S20):

{section_block}

For EACH of the topics listed below, identify which sections contain the rule(s)
relevant to answering that topic in {country_name}. Output a JSON object whose
keys are the topic names and whose values are LISTS of section IDs (e.g.
["S4","S15"]). If no section in this country's handbook covers the topic, return
an empty list `[]` for that topic.

Topics (with a representative question for each):
{topic_block}

Rules:
- Only use section IDs that EXIST in the handbook above.
- Prefer the section whose summary directly addresses the topic.
- Multiple sections allowed (max 3) when more than one is relevant.
- Empty list `[]` is acceptable when this country has no relevant section.

Output STRICT JSON (no markdown, no commentary):
{{
  "topic_a": ["Sxx"],
  "topic_b": [...],
  ...
}}
"""


def gpt(cli, prompt, model='gpt-5.4', reasoning_effort='medium', retries=3):
    backoff = 3.0
    last_raw = ''
    for _ in range(retries):
        try:
            r = cli.chat.completions.create(
                model=model,
                messages=[{'role':'user','content':prompt}],
                max_completion_tokens=4000,
                reasoning_effort=reasoning_effort,
                timeout=180)
            t = (r.choices[0].message.content or '').strip()
            last_raw = t
            t = re.sub(r'^```(?:json)?\s*|\s*```$', '', t, flags=re.S).strip()
            return json.loads(t)
        except json.JSONDecodeError:
            return {'error': 'json_decode', 'raw': last_raw[:300]}
        except Exception as e:
            time.sleep(backoff); backoff = min(backoff*2, 30)
    return {'error': 'retries_exhausted'}


def collect_topic_questions():
    """Get one representative question per template_topic from v2 region items."""
    v2 = json.load(open(REPO / 'culturebenchmark_eval_v2.json'))
    region = [r for r in v2 if r.get('question_category') == 'region']
    seen = {}
    for r in region:
        t = r['template_topic']
        if t not in seen:
            # strip the verbose two-step prefix for brevity
            q = r['question']
            if 'answer the following question' in q:
                q = q.split('answer the following question:')[-1].strip(' :\n')
            seen[t] = q[:200]
    return seen


def build_mapping(cli, country, topic_questions):
    handbook = load_handbook(country)
    secs = section_titles(handbook)
    section_block = '\n'.join(f'  {sid}: {title}' for sid, title in secs)
    topic_block = '\n'.join(f'  {t}:  Q: {q}' for t, q in sorted(topic_questions.items()))
    prompt = PROMPT.format(
        country_name=COUNTRIES[country],
        section_block=section_block,
        topic_block=topic_block,
    )
    return gpt(cli, prompt)


def audit_mapping(mapping, topic_questions):
    """Quick text-based sanity check of mapping (no GPT)."""
    # Print per-country topic→sections mapping
    countries = sorted(mapping.keys())
    topics = sorted(topic_questions.keys())
    print(f'\n{"topic":<45s} | ' + ' | '.join(f'{c:<8s}' for c in countries))
    print('-' * (47 + 11 * len(countries)))
    for t in topics:
        cells = []
        for c in countries:
            cell = mapping[c].get(t, [])
            cells.append(','.join(cell) if cell else '∅')
        print(f'{t:<45s} | ' + ' | '.join(f'{c:<8s}' for c in cells))


def apply_to_benchmark(mapping, bench_path, dry_run=False):
    bench = json.load(open(bench_path))
    n_updated = n_unmapped = 0
    unmapped = set()
    for r in bench:
        if r.get('question_category') != 'region': continue
        c = r['country']
        t = r.get('template_topic')
        refs = mapping.get(c, {}).get(t, [])
        if not refs:
            n_unmapped += 1
            unmapped.add((c, t))
        r['rule_reference'] = list(refs)
        n_updated += 1
    if not dry_run:
        json.dump(bench, open(bench_path, 'w'), ensure_ascii=False, indent=2)
    return {'updated': n_updated, 'unmapped': n_unmapped,
            'unmapped_pairs': sorted(unmapped)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(REPO / 'eval' / 'analysis' / 'region_per_country_mapping.json'))
    ap.add_argument('--apply', action='store_true', help='apply mapping to v2 JSON')
    ap.add_argument('--bench', default=str(REPO / 'culturebenchmark_eval_v2.json'))
    args = ap.parse_args()

    cli = OpenAI(api_key=os.environ['OPENAI_API_KEY'].strip())
    topic_qs = collect_topic_questions()
    print(f'Topics: {len(topic_qs)}')

    mapping = {}
    print('Building per-country mapping (6 GPT-5.4 calls) ...')
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = {pool.submit(build_mapping, cli, c, topic_qs): c for c in COUNTRIES}
        for fut in as_completed(futs):
            c = futs[fut]
            res = fut.result()
            if isinstance(res, dict) and 'error' not in res:
                mapping[c] = res
                print(f'  ✅ {c}: {len(res)} topics mapped')
            else:
                print(f'  ❌ {c}: {res}')
                mapping[c] = {}

    json.dump(mapping, open(args.out, 'w'), ensure_ascii=False, indent=2)
    print(f'\nSaved per-country mapping → {args.out}')

    audit_mapping(mapping, topic_qs)

    if args.apply:
        res = apply_to_benchmark(mapping, args.bench)
        print(f'\nApply: updated={res["updated"]}, unmapped={res["unmapped"]}')
        if res['unmapped_pairs']:
            print('  unmapped pairs (no section in that country):')
            for c, t in res['unmapped_pairs']:
                print(f'    {c}/{t}')


if __name__ == '__main__':
    main()
