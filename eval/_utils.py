"""Shared utilities across inference and analysis scripts."""
import json
import os
import re
from typing import Any, Dict, Iterable, List, Tuple

# --- benchmark constants ---------------------------------------------------
COUNTRIES = ['cn', 'us', 'uk', 'jp', 'sg', 'ind']
COUNTRY_LABELS = {'cn': 'CN', 'us': 'US', 'uk': 'UK',
                  'jp': 'JP', 'sg': 'SG', 'ind': 'IND'}
CATEGORIES = ['perception', 'prediction', 'planning', 'region']
CATEGORY_LABELS = {'perception': 'Perc.', 'prediction': 'Pred.',
                   'planning': 'Plan.',  'region': 'Reg.'}
SETTINGS = ['direct', 'reasoning', 'rule_given']
SETTING_LABELS = {'direct': 'Direct', 'reasoning': 'Reasoning',
                  'rule_given': 'Rule-Given'}

# Country palette used by all plots (user-specified).
# Index order must match COUNTRIES above.
PALETTE = {
    'cn':  '#E8706F',  # salmon red
    'us':  '#F0D258',  # mustard yellow
    'uk':  '#678CB5',  # steel blue
    'jp':  '#F0973B',  # orange
    'sg':  '#87C1BD',  # teal
    'ind': '#6DB066',  # soft green
}

# --- JSON helpers ----------------------------------------------------------
def load_json(path: str) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(obj: Any, path: str) -> None:
    """Atomic write: write to .tmp then rename, so partial writes never
    corrupt the result file (which would break resume on restart)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

# --- composite key for de-dup / resume ------------------------------------
def composite_key(r: Dict) -> Tuple:
    """Resume key: just `id`. We assume the benchmark file is the source of
    truth; matching by id alone makes resume robust to harmless edits in the
    saved record (e.g. question text / image_path normalisation drift).

    If you need finer-grained dedup (counterfactual variants in the same
    benchmark), use `composite_key_full`.
    """
    return (r.get('id'),)

def composite_key_full(r: Dict) -> Tuple:
    """Full key (id, image_path, question) — kept for backward compatibility
    when truly de-duplicating across benchmark variants."""
    img = r.get('image_path') or []
    return (r.get('id'),
            tuple(img) if isinstance(img, list) else (img,),
            r.get('question', ''))

# --- answer extraction from model output ----------------------------------
def extract_ans(pred: str):
    """Pull a single letter (A-D) from a model output — robust to verbose CoT.

    Order of preference:
      1. The LAST `final answer: X` (so a recap line wins over an earlier
         "Step 4 — Answer:" header).
      2. The LAST `answer: X` where X is followed by a non-letter (so
         "Answer: Based on..." won't match "B" from "Based").
      3. The last A-D word-bounded letter in the final 60 chars.
      4. The last A-D word-bounded letter anywhere.
    """
    s = (pred or '').strip()
    if not s:
        return None
    ms = re.findall(r'final\s*answer\s*[:：]\s*([A-D])\b', s, re.I)
    if ms:
        return ms[-1].upper()
    ms = re.findall(r'answer\s*[:：]\s*([A-D])(?=[^A-Za-z]|$)', s, re.I)
    if ms:
        return ms[-1].upper()
    tail = s[-60:]
    ms = re.findall(r'\b([A-D])\b', tail)
    if ms:
        return ms[-1]
    ms = re.findall(r'\b([A-D])\b', s)
    return ms[-1] if ms else None

# --- resume loader (shared across inference scripts) ----------------------
def load_with_resume(output_path: str, overwrite: bool = False) -> Tuple[List[Dict], set]:
    """Load existing result JSON, drop ERROR entries so they retry, return
    (results, done_keys).

    Robust to:
      - Missing file → start fresh
      - Corrupt JSON (e.g. partial write before atomic-rename was added) →
        also try loading from a `.tmp` sibling, otherwise warn and start fresh
        WITHOUT deleting the broken file (so user can inspect).
    """
    if overwrite:
        print(f'[resume] --overwrite given; ignoring existing {output_path}')
        return [], set()
    if not os.path.exists(output_path):
        return [], set()

    try:
        results = load_json(output_path)
    except json.JSONDecodeError as e:
        # Atomic save now uses .tmp; on crash partway, .tmp may be the only
        # complete copy. Try it before declaring loss.
        tmp = output_path + '.tmp'
        if os.path.exists(tmp):
            try:
                results = load_json(tmp)
                print(f'[resume] main file corrupt, loaded from {tmp}')
            except Exception:
                print(f'[resume] WARNING: {output_path} is corrupt JSON ({e}). '
                      f'Both file and .tmp unreadable. Refusing to overwrite — '
                      f'aborting. Pass --overwrite to start fresh.')
                raise
        else:
            print(f'[resume] WARNING: {output_path} is corrupt JSON ({e}). '
                  f'Refusing to overwrite — aborting. '
                  f'Pass --overwrite to start fresh.')
            raise

    pre = len(results)
    results = [r for r in results if not str(r.get('pred', '')).startswith('ERROR')]
    dropped = pre - len(results)
    if dropped:
        print(f'[resume] Dropped {dropped} prior ERROR entries for retry.')
    done_keys = {composite_key(r) for r in results}
    print(f'[resume] {len(done_keys)} items already done in {output_path}.')
    return results, done_keys
