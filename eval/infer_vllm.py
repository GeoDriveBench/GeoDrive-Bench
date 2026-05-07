"""Multi-backend inference for CultureBenchmark local VLMs.

Three backends share one loop / one registry:
    vllm      — vLLM continuous batching   (llava, qwen25vl, qwen3vl, internvl3, gemma3)
    hf        — HuggingFace transformers   (llama32)
    lmdeploy  — lmdeploy TurboMind         (internvl35)

Example
    python infer_vllm.py --model qwen3vl --setting direct
    python infer_vllm.py --model internvl35 --setting rule_given
"""

import argparse, os, re, sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from _utils import composite_key, load_json, save_json, load_with_resume

# Handbooks (from repo-root `traffic_handbook/`) drive the rule_given prompt.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from traffic_handbook import (
    china_traffic_handbook, india_traffic_handbook, japan_traffic_handbook,
    singapore_traffic_handbook, uk_traffic_handbook, us_traffic_handbook,
)
HANDBOOKS = {
    "cn": china_traffic_handbook, "ind": india_traffic_handbook,
    "jp": japan_traffic_handbook, "sg": singapore_traffic_handbook,
    "uk": uk_traffic_handbook,    "us": us_traffic_handbook,
}

#
# DriveOPD checkpoint paths are read from env vars so users can plug in
# their own training output without editing this file.
#   DRIVEOPD_QWEN25VL_PATH   — path to the SDFT-tuned Qwen2.5-VL-7B checkpoint
#   DRIVEOPD_INTERNVL3_PATH  — path to the SDFT-tuned InternVL3-8B checkpoint
import os

MODEL_REGISTRY: Dict[str, Dict] = {
    "llava":      {"path": "llava-hf/llava-v1.6-mistral-7b-hf",       "engine": "vllm",     "max_images": 1},
    "qwen25vl":   {"path": "Qwen/Qwen2.5-VL-7B-Instruct",             "engine": "vllm",     "max_images": 6},
    "qwen3vl":    {"path": "Qwen/Qwen3-VL-8B-Instruct",               "engine": "vllm",     "max_images": 6, "enable_thinking": False},
    "internvl3":  {"path": "OpenGVLab/InternVL3-8B",                  "engine": "vllm",     "max_images": 6},
    "internvl35": {"path": "OpenGVLab/InternVL3_5-8B",                "engine": "vllm",     "max_images": 6},
    "gemma3":     {"path": "google/gemma-3-12b-it",                   "engine": "vllm",     "max_images": 6},
    "llama32":    {"path": "meta-llama/Llama-3.2-11B-Vision-Instruct","engine": "hf",       "max_images": 1},
    # DriveOPD (Ours) — set env vars to point at your SDFT checkpoints.
    "driveopd_qwen25vl":  {"path": os.environ.get("DRIVEOPD_QWEN25VL_PATH",  "PATH/TO/qwen25vl_sdft/checkpoint"),
                           "engine": "vllm", "max_images": 6},
    "driveopd_internvl3": {"path": os.environ.get("DRIVEOPD_INTERNVL3_PATH", "PATH/TO/internvl3_sdft/checkpoint"),
                           "engine": "vllm", "max_images": 6},
}

# --------------------------------------------------------------- prompts ----
_COUNTRY_PATTERNS = [
    (r"\b(?:In|in)\s+mainland\s+China\b", "in this country"),
    (r"\b(?:In|in)\s+(?:Japan|the\s+UK|Singapore|India|the\s+United\s+Kingdom|the\s+United\s+States|the\s+US|the\s+USA|China|Japan|Singapore|India)\b", "in this country"),
    (r"\b(?:mainland\s+China|United\s+Kingdom|United\s+States|USA)\b", "this country"),
    (r"\b(?:Chinese|Japanese|British|Singaporean|Indian|American)\b", "local"),
    (r"\bChina\b", "this country"),
    (r"\bJapan\b", "this country"),
    (r"\bSingapore\b", "this country"),
    (r"\bIndia\b", "this country"),
    (r"\bthe\s+UK\b", "this country"),
    (r"\bthe\s+US\b", "this country"),
]

def anonymize_country(text: str) -> str:
    """Remove country-specific references from handbook/rule text."""
    for pat, repl in _COUNTRY_PATTERNS:
        text = re.sub(pat, repl, text)
    return text

def extract_rules(handbook: str, refs: List[str]) -> str:
    out = []
    for ref in refs or []:
        m = re.search(rf"({re.escape(ref)}\s.+?)(?=\nS\d+\s|\Z)", handbook, re.DOTALL)
        if m:
            out.append(m.group(1).strip()); continue
        for line in handbook.splitlines():
            if line.startswith(ref + " ") or line.startswith(ref + ":"):
                out.append(line.strip()); break
    return "\n\n".join(out)

_PROMPTS = {
    "direct": (
        "Answer the question using only the image. Choose exactly one option.\n"
        "Question: {q}\nOptions:\n{opts}\n"
        "Output only the answer letter (A/B/C/D).\n"
    ),
    "reasoning": (
        "Step 1 — Geographic inference: Based on road infrastructure, traffic signs, "
        "lane markings, and environmental cues in the image, identify the most likely "
        "country or region where this scene was captured.\n"
        "Step 2 — Cultural traffic rule: State the relevant traffic rule(s) for that "
        "country that apply to this question.\n"
        "Step 3 — Visual reasoning: Describe what you observe in the image that is "
        "relevant to answering the question.\n"
        "Step 4 — Answer: Choose exactly one option.\n"
        "Question: {q}\nOptions:\n{opts}\n"
        "Output your step-by-step reasoning followed by the final answer letter (A/B/C/D).\n"
    ),
    "rule_given": (
        "{rule_block}"
        "Using the traffic rule(s) above and the image, answer the question by following these steps:\n"
        "Step 1 — Rule understanding: Identify which part of the given traffic rule is relevant to this question.\n"
        "Step 2 — Visual observation: Describe what you observe in the image that relates to the question.\n"
        "Step 3 — Rule application: Apply the traffic rule to the observed scene to reason about the correct answer.\n"
        "Step 4 — Answer: Choose exactly one option.\n\n"
        "Question: {q}\nOptions:\n{opts}\n"
        "Output your step-by-step reasoning followed by the final answer letter (A/B/C/D).\n"
    ),
    "full_handbook": (
        "{rule_block}"
        "Using the traffic handbook above and the image, answer the question by following these steps:\n"
        "Step 1 — Rule understanding: Identify which rules in the handbook are relevant to this question.\n"
        "Step 2 — Visual observation: Describe what you observe in the image that relates to the question.\n"
        "Step 3 — Rule application: Apply the relevant rule(s) to the observed scene to reason about the correct answer.\n"
        "Step 4 — Answer: Choose exactly one option.\n\n"
        "Question: {q}\nOptions:\n{opts}\n"
        "Output your step-by-step reasoning followed by the final answer letter (A/B/C/D).\n"
    ),
    "wrong_rule": (
        "{rule_block}"
        "Using the traffic rule(s) above and the image, answer the question by following these steps:\n"
        "Step 1 — Rule understanding: Identify which part of the given traffic rule is relevant to this question.\n"
        "Step 2 — Visual observation: Describe what you observe in the image that relates to the question.\n"
        "Step 3 — Rule application: Apply the traffic rule to the observed scene to reason about the correct answer.\n"
        "Step 4 — Answer: Choose exactly one option.\n\n"
        "Question: {q}\nOptions:\n{opts}\n"
        "Output your step-by-step reasoning followed by the final answer letter (A/B/C/D).\n"
    ),
    "no_image": (
        "Answer the question using only your general knowledge (no image is provided). "
        "Choose exactly one option.\n"
        "Question: {q}\nOptions:\n{opts}\n"
        "Output only the answer letter (A/B/C/D).\n"
    ),
    # image_corruption uses the same prompt as `direct`; only the pixels differ.
    "image_corruption": (
        "Answer the question using only the image. Choose exactly one option.\n"
        "Question: {q}\nOptions:\n{opts}\n"
        "Output only the answer letter (A/B/C/D).\n"
    ),
    # Reasoning-CoT variants of the two robustness settings, to study whether
    # multi-step CoT amplifies hallucination when visual info is missing/degraded.
    "no_image_reasoning": (
        "Step 1 — Geographic inference: Based on road infrastructure, traffic signs, "
        "lane markings, and environmental cues in the image, identify the most likely "
        "country or region where this scene was captured.\n"
        "Step 2 — Cultural traffic rule: State the relevant traffic rule(s) for that "
        "country that apply to this question.\n"
        "Step 3 — Visual reasoning: Describe what you observe in the image that is "
        "relevant to answering the question.\n"
        "Step 4 — Answer: Choose exactly one option.\n"
        "Question: {q}\nOptions:\n{opts}\n"
        "Output your step-by-step reasoning followed by the final answer letter (A/B/C/D).\n"
    ),
    "image_corruption_reasoning": (
        "Step 1 — Geographic inference: Based on road infrastructure, traffic signs, "
        "lane markings, and environmental cues in the image, identify the most likely "
        "country or region where this scene was captured.\n"
        "Step 2 — Cultural traffic rule: State the relevant traffic rule(s) for that "
        "country that apply to this question.\n"
        "Step 3 — Visual reasoning: Describe what you observe in the image that is "
        "relevant to answering the question.\n"
        "Step 4 — Answer: Choose exactly one option.\n"
        "Question: {q}\nOptions:\n{opts}\n"
        "Output your step-by-step reasoning followed by the final answer letter (A/B/C/D).\n"
    ),
    # rule_given × no-image / corrupted-image variants — prompt is identical to
    # `rule_given`; only the pixels (or absence thereof) differ at runtime.
    "no_image_rule_given": (
        "{rule_block}"
        "Using the traffic rule(s) above and the image, answer the question by following these steps:\n"
        "Step 1 — Rule understanding: Identify which part of the given traffic rule is relevant to this question.\n"
        "Step 2 — Visual observation: Describe what you observe in the image that relates to the question.\n"
        "Step 3 — Rule application: Apply the traffic rule to the observed scene to reason about the correct answer.\n"
        "Step 4 — Answer: Choose exactly one option.\n\n"
        "Question: {q}\nOptions:\n{opts}\n"
        "Output your step-by-step reasoning followed by the final answer letter (A/B/C/D).\n"
    ),
    "image_corruption_rule_given": (
        "{rule_block}"
        "Using the traffic rule(s) above and the image, answer the question by following these steps:\n"
        "Step 1 — Rule understanding: Identify which part of the given traffic rule is relevant to this question.\n"
        "Step 2 — Visual observation: Describe what you observe in the image that relates to the question.\n"
        "Step 3 — Rule application: Apply the traffic rule to the observed scene to reason about the correct answer.\n"
        "Step 4 — Answer: Choose exactly one option.\n\n"
        "Question: {q}\nOptions:\n{opts}\n"
        "Output your step-by-step reasoning followed by the final answer letter (A/B/C/D).\n"
    ),
}

_ALL_RULE_IDS = {}  # country -> list of rule IDs

def _get_rule_ids(country: str) -> List[str]:
    if country not in _ALL_RULE_IDS:
        handbook = HANDBOOKS.get(country, "")
        ids = re.findall(r"^(S\d+)\s", handbook, re.MULTILINE)
        _ALL_RULE_IDS[country] = ids
    return _ALL_RULE_IDS[country]

def build_prompt(item: Dict, setting: str) -> str:
    import random
    q, options = item["question"], item["options"]
    opts = "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(options))
    country = item["country"]
    handbook = HANDBOOKS.get(country, "")

    if setting in ("rule_given", "no_image_rule_given", "image_corruption_rule_given"):
        rule = extract_rules(handbook, item.get("rule_reference", []))
        rule = anonymize_country(rule) if rule else ""
        rule_block = f"Relevant traffic rule(s):\n{rule}\n\n" if rule else ""
        return _PROMPTS[setting].format(q=q, opts=opts, rule_block=rule_block)

    if setting == "full_handbook":
        anon = anonymize_country(handbook) if handbook else ""
        rule_block = f"Traffic handbook:\n{anon}\n\n" if anon else ""
        return _PROMPTS[setting].format(q=q, opts=opts, rule_block=rule_block)

    if setting == "wrong_rule":
        # Pick a random rule NOT in rule_reference
        correct_refs = set(item.get("rule_reference", []))
        all_ids = _get_rule_ids(country)
        candidates = [r for r in all_ids if r not in correct_refs]
        rand = random.Random(item.get("id", 0))  # deterministic per item
        if candidates:
            picked = [rand.choice(candidates)]
            rule = extract_rules(handbook, picked)
            rule = anonymize_country(rule) if rule else ""
            rule_block = f"Relevant traffic rule(s):\n{rule}\n\n" if rule else ""
        else:
            rule_block = ""
        return _PROMPTS[setting].format(q=q, opts=opts, rule_block=rule_block)

    return _PROMPTS[setting].format(q=q, opts=opts)

# ---------------------------------------------------------------- image ----
def _img(path: str) -> Image.Image:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")


# ImageNet-C-style severities (sev ∈ 1..5, higher = worse).
_CORRUPTION_PARAMS = {
    "gaussian_noise": [10, 25, 50, 75, 100],   # σ in 0-255 space
    "blur":           [2,  4,  6,  9,  14],    # gaussian blur radius
    "pixelate":       [0.6, 0.4, 0.3, 0.2, 0.1],  # downsample fraction
    "jpeg":           [25, 18, 15, 10, 7],     # JPEG quality
}


def _corrupt_image(img: "Image.Image", kind: str, sev: int) -> "Image.Image":
    import io
    import numpy as np
    from PIL import ImageFilter
    if kind not in _CORRUPTION_PARAMS:
        return img
    p = _CORRUPTION_PARAMS[kind][max(1, min(5, sev)) - 1]
    if kind == "gaussian_noise":
        rng = np.random.RandomState(42)  # deterministic
        arr = np.asarray(img, dtype=np.float32)
        arr = arr + rng.normal(0, p, arr.shape).astype(np.float32)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=p))
    if kind == "pixelate":
        w, h = img.size
        small = img.resize((max(1, int(w * p)), max(1, int(h * p))), Image.NEAREST)
        return small.resize((w, h), Image.NEAREST)
    if kind == "jpeg":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=p)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    return img


def _resolve_images(paths, cfg):
    """Centralized image loading honoring no_image / image_corruption settings."""
    setting = cfg.get("setting", "")
    if setting in ("no_image", "no_image_reasoning", "no_image_rule_given"):
        return []
    imgs = [_img(p) for p in paths]
    if setting in ("image_corruption", "image_corruption_reasoning", "image_corruption_rule_given"):
        kind = cfg.get("corruption_type", "gaussian_noise")
        sev  = cfg.get("corruption_severity", 3)
        imgs = [_corrupt_image(im, kind, sev) for im in imgs]
    return imgs

# ---------------------------------------------------------------- vLLM  ----
def _load_vllm(cfg: Dict):
    from vllm import LLM
    from transformers import AutoProcessor, AutoTokenizer
    # InternVL3.5's processor in current transformers has a missing-attribute
    # bug (Qwen2TokenizerFast has no `start_image_token`). Fall back to the
    # tokenizer alone — its `apply_chat_template` is sufficient since vLLM
    # handles the multi-modal token expansion internally via `multi_modal_data`.
    try:
        proc = AutoProcessor.from_pretrained(cfg["path"], trust_remote_code=True)
    except AttributeError as e:
        if "start_image_token" in str(e):
            proc = AutoTokenizer.from_pretrained(cfg["path"], trust_remote_code=True)
        else:
            raise
    llm = LLM(
        model=cfg["path"], dtype="bfloat16", max_model_len=16384,
        limit_mm_per_prompt={"image": cfg["max_images"]},
        trust_remote_code=True, gpu_memory_utilization=0.90,
        max_num_seqs=64, disable_mm_preprocessor_cache=True,
        mm_processor_cache_gb=0, enforce_eager=True,
    )
    return proc, llm

def _call_vllm(proc, llm, cfg, batch: List[Tuple[List[str], str]], max_tokens: int):
    from vllm import SamplingParams
    reqs = []
    for imgs, prompt in batch:
        images = _resolve_images(imgs, cfg)
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]
        kwargs = {"enable_thinking": False} if cfg.get("enable_thinking") is False else {}
        text = proc.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True, **kwargs,
        )
        if images:
            reqs.append({"prompt": text, "multi_modal_data": {"image": images}})
        else:
            reqs.append({"prompt": text})
    out = llm.generate(reqs, sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0))
    return [o.outputs[0].text.strip() for o in out]

# ------------------------------------------------------------- HuggingFace -
def _load_hf(cfg: Dict):
    from transformers import AutoProcessor, MllamaForConditionalGeneration
    proc = AutoProcessor.from_pretrained(cfg["path"])
    mdl = MllamaForConditionalGeneration.from_pretrained(
        cfg["path"], torch_dtype=torch.bfloat16, device_map="auto").eval()
    return proc, mdl

def _call_hf(proc, mdl, cfg, batch, max_tokens):
    preds = []
    for imgs, prompt in batch:
        images = [_img(p) for p in imgs]
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]
        text = proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True)
        ins = proc(text=text, images=images, return_tensors="pt").to(mdl.device)
        with torch.no_grad():
            gen = mdl.generate(**ins, max_new_tokens=max_tokens, do_sample=False)
        preds.append(proc.decode(gen[0][ins["input_ids"].shape[1]:], skip_special_tokens=True).strip())
    return preds

# ----------------------------------------------------------------- lmdeploy
def _load_lmdeploy(cfg: Dict):
    from lmdeploy import ChatTemplateConfig, TurbomindEngineConfig, pipeline
    pipe = pipeline(
        cfg["path"],
        backend_config=TurbomindEngineConfig(session_len=16384, tp=1, device="cuda"),
        chat_template_config=ChatTemplateConfig(
            model_name=cfg.get("chat_template", "internvl2_5")),
    )
    return None, pipe

def _call_lmdeploy(_proc, pipe, cfg, batch, max_tokens):
    from lmdeploy.vl import load_image as lmd_load
    preds = []
    for imgs, prompt in batch:
        if len(imgs) > 1:
            images = [lmd_load(p) for p in imgs]
            pfx = "".join(f"Frame{i+1}: <image>\n" for i in range(len(images)))
            resp = pipe((pfx + prompt, images))
        else:
            resp = pipe((prompt, lmd_load(imgs[0])))
        preds.append(resp.text)
    return preds

ENGINES = {
    "vllm":     (_load_vllm,     _call_vllm),
    "hf":       (_load_hf,       _call_hf),
    "lmdeploy": (_load_lmdeploy, _call_lmdeploy),
}

# ---------------------------------------------------------------- runner ---
def run(args):
    cfg = dict(MODEL_REGISTRY[args.model])
    if args.model_path:
        cfg["path"] = args.model_path
    # carry runtime flags into cfg so engine fns can branch on setting
    cfg["setting"] = args.setting
    cfg["corruption_type"] = args.corruption_type
    cfg["corruption_severity"] = args.corruption_severity
    print(f"Loading {args.model} [{cfg['path']}] via {cfg['engine']} ...")
    load, call = ENGINES[cfg["engine"]]
    proc, mdl = load(cfg)

    data = load_json(args.benchmark)
    if args.country:  data = [d for d in data if d["country"] in args.country]
    if args.category: data = [d for d in data if d["question_category"] in args.category]

    results, done_keys = load_with_resume(args.output, args.overwrite)
    pending = [it for it in data if composite_key(it) not in done_keys]
    print(f"Pending: {len(pending)} / {len(data)}")
    if not pending:
        save_json(results, args.output); return

    n_tokens = args.max_new_tokens if args.setting == "direct" else args.max_new_tokens_long
    max_img = cfg["max_images"]
    batched = cfg["engine"] == "vllm"
    BATCH = args.batch_size if batched else 1

    def _pack(items):
        return [((it["image_path"][-max_img:] if len(it["image_path"]) > max_img
                  else it["image_path"]), build_prompt(it, args.setting))
                for it in items]

    def _save_row(it, pred):
        results.append({
            "id": it["id"], "country": it["country"],
            "question_category": it["question_category"],
            "question_type": it.get("question_type", "multiple_choice"),
            "image_path": it["image_path"], "question": it["question"],
            "options": it["options"], "gt": it["answer"], "pred": pred,
            "rule_reference": it.get("rule_reference", []),
            "setting": args.setting,
        })

    save_every = max(50, BATCH)  # rows between disk syncs
    since_save = 0
    for i in tqdm(range(0, len(pending), BATCH), desc=f"{args.model}/{args.setting}"):
        chunk = pending[i:i + BATCH]
        reqs = _pack(chunk)
        try:
            preds = call(proc, mdl, cfg, reqs, n_tokens)
        except Exception:
            # Engine crashed: retry per-item so one bad sample doesn't kill the chunk.
            preds = []
            for r in reqs:
                try:
                    preds.extend(call(proc, mdl, cfg, [r], n_tokens))
                except Exception as ee:
                    preds.append(f"ERROR: {ee}")
        for it, pred in zip(chunk, preds):
            _save_row(it, pred)
        since_save += len(chunk)
        if since_save >= save_every:
            save_json(results, args.output); since_save = 0

    save_json(results, args.output)
    print(f"Saved {len(results)} results → {args.output}")


def main():
    ROOT = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description="CultureBenchmark VLM inference")
    p.add_argument("--model",    required=True, choices=list(MODEL_REGISTRY.keys()))
    p.add_argument("--setting",  required=True, choices=[
        "direct", "reasoning", "rule_given", "full_handbook", "wrong_rule",
        "no_image", "image_corruption",
        "no_image_reasoning", "image_corruption_reasoning",
        "no_image_rule_given", "image_corruption_rule_given",
    ])
    p.add_argument("--corruption_type", default="gaussian_noise",
                   choices=list(_CORRUPTION_PARAMS.keys()),
                   help="image_corruption only")
    p.add_argument("--corruption_severity", type=int, default=3, choices=[1, 2, 3, 4, 5],
                   help="image_corruption only (1=mild, 5=severe)")
    p.add_argument("--model_path", default=None)
    p.add_argument("--benchmark",  default=str(ROOT / "culturebenchmark_eval.json"))
    p.add_argument("--output",     default=None)
    p.add_argument("--max_new_tokens",      type=int, default=32)
    p.add_argument("--max_new_tokens_long", type=int, default=512)
    p.add_argument("--batch_size",          type=int, default=64)
    p.add_argument("--country",  nargs="+", default=None)
    p.add_argument("--category", nargs="+", default=None)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    if a.output is None:
        a.output = str(Path(__file__).resolve().parent / "results" /
                       f"{a.model}_{a.setting}_results.json")
    run(a)


if __name__ == "__main__":
    main()
