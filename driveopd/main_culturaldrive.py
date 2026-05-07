"""On-policy Self-Distillation on CulturalDrive.

Student sees the `reasoning` prompt (4-step CoT; no traffic rule).
Teacher sees the `rule_given`  prompt (4-step CoT; rule provided).
Both prompt styles expect a multi-token CoT completion, which gives the
KL loss a long supervision signal (as opposed to the single-letter "direct"
response). The student learns to internalise the rule via distillation.

Smoke test:
    python main_culturaldrive.py --smoke --output_dir /tmp/distil_smoke

Full run:
    python main_culturaldrive.py --output_dir ./runs/qwen3vl-8b-sdft
"""
import argparse
import json
import os
import sys

import torch
import transformers
from datasets import Dataset
from PIL import Image
from transformers import AutoConfig, AutoProcessor

from distil_config import DistilConfig
from distil_trainer import DistilTrainer

# Reuse the *identical* prompt builders we used at eval time so the
# student/teacher distributions match the evaluation protocol 1:1.
_EVAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "eval"))
sys.path.insert(0, _EVAL_DIR)
from infer_vllm import build_prompt  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name",   default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--dataset_path", default=os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "culturebenchmark_eval.json")))
    p.add_argument("--output_dir",   required=True)

    p.add_argument("--learning_rate",  type=float, default=1e-5)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--num_prompts_per_batch", type=int, default=16,
                   help="= gradient_accumulation_steps")
    p.add_argument("--ref_model_mixup_alpha", type=float, default=0.01)

    p.add_argument("--max_train_samples",    type=int, default=None)
    # Qwen3-VL encodes each 1600×900 image as ~1400 tokens, so multi-image
    # prompts easily blow past 4k tokens. Default to 1 image for training.
    p.add_argument("--max_images",           type=int, default=1)
    p.add_argument("--max_prompt_length",    type=int, default=2560)
    p.add_argument("--max_completion_length", type=int, default=512)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true",
                   help="16 samples, 1 micro-batch, no wandb, no save")
    p.add_argument("--dry_run", action="store_true",
                   help="Build dataset + models, then exit before training")
    return p.parse_args()


def _load_image(path: str) -> Image.Image:
    """Resize to fit InternVL's single 448×448 tile (256 visual tokens) so
    image-token count is consistent across models. Qwen2.5-VL handles arbitrary
    sizes but still benefits from a small input."""
    img = Image.open(path).convert("RGB")
    return img.resize((448, 448), Image.BILINEAR)


def load_culturaldrive(path: str, *, max_samples=None, max_images=3, seed=42) -> Dataset:
    """Build the SDFT dataset.

    Text prompts are materialised once up-front (fast). Image decoding is
    *lazy* via `Dataset.set_transform`: each __getitem__ reads PIL images
    from disk on demand, so 11k samples × 4 ranks don't all preload.
    """
    raw = json.load(open(path))
    if max_samples is not None:
        raw = raw[:max_samples]

    def _paths(ex):
        imgs = ex["image_path"]
        if len(imgs) > max_images:
            imgs = imgs[-max_images:]
        return imgs

    rows = [
        {
            "prompt":         [{"role": "user", "content": build_prompt(ex, "reasoning")}],
            # Teacher gets the FULL country handbook (not just the cited rule).
            # Forces it to locate the relevant part itself — harder / more honest.
            "teacher_prompt": [{"role": "user", "content": build_prompt(ex, "full_handbook")}],
            "image_paths":    _paths(ex),
        }
        for ex in raw
    ]
    ds = Dataset.from_list(rows)

    def _lazy_images(batch):
        batch["images"] = [[_load_image(p) for p in paths] for paths in batch["image_paths"]]
        return batch

    ds.set_transform(_lazy_images)
    ds = ds.shuffle(seed=seed)
    print(f"[data] {len(ds)} items (lazy image load, max_images={max_images})")
    return ds


def main():
    args = parse_args()
    if args.smoke:
        if args.max_train_samples is None:
            args.max_train_samples = 16
        args.num_prompts_per_batch = min(args.num_prompts_per_batch, 4)
        args.output_dir = args.output_dir or "/tmp/sdft_smoke"
        print(f"[smoke] samples={args.max_train_samples} "
              f"accum={args.num_prompts_per_batch} output={args.output_dir}")

    # VLM auto-detect (AutoConfig.architectures[0] — e.g. Qwen3VLForConditionalGeneration).
    # Fall back to AutoModel + trust_remote_code for VLMs whose class lives in
    # the model repo (e.g. InternVL3's `InternVLChatModel`).
    cfg = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    arch_name = cfg.architectures[0]
    arch = getattr(transformers, arch_name, None)
    if arch is None:
        from transformers import AutoModel
        arch = AutoModel
        load_kwargs = {"torch_dtype": torch.bfloat16, "trust_remote_code": True}
    else:
        load_kwargs = {"torch_dtype": torch.bfloat16}
    print(f"[model] {args.model_name} → {arch.__name__} (arch_name={arch_name})")

    model   = arch.from_pretrained(args.model_name, **load_kwargs)
    teacher = arch.from_pretrained(args.model_name, **load_kwargs)
    processor = AutoProcessor.from_pretrained(args.model_name, truncation_side="left", trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    dataset = load_culturaldrive(
        args.dataset_path,
        max_samples=args.max_train_samples,
        max_images=args.max_images,
        seed=args.seed,
    )

    if args.dry_run:
        print("[dry-run] data + models loaded OK — exiting before trainer.train()")
        return

    dc = DistilConfig(
        seed=args.seed,
        # vLLM colocation for on-policy generation
        use_vllm=True, vllm_mode="colocate", vllm_tensor_parallel_size=1,
        vllm_gpu_memory_utilization=0.4, vllm_enable_sleep_mode=True,
        # optimisation
        learning_rate=args.learning_rate, warmup_ratio=0.1,
        lr_scheduler_type="cosine", logging_steps=1,
        bf16=True, fp16=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.num_prompts_per_batch,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_train_epochs=args.num_train_epochs,
        num_iterations=1, num_generations=1,
        save_steps=100 if not args.smoke else 10**9,
        max_grad_norm=1.0,
        report_to="wandb",   # always log to wandb
        run_name=(f"qwen3vl8b_sdft_smoke" if args.smoke else "qwen3vl8b_sdft"),
        output_dir=args.output_dir,
        log_completions=args.smoke,
        # Teacher is a fixed rule-conditioned oracle — do NOT sync it with
        # the student. This also removes the per-step ~10-30s parameter-wise
        # gather/mixup + the extra vLLM weight sync.
        sync_ref_model=False,
        ref_model_mixup_alpha=args.ref_model_mixup_alpha,
        # Skip the extra student forward used to correct vLLM/HF log-prob drift.
        # It doubles student forward cost for marginal statistical benefit.
        vllm_importance_sampling_correction=False,
        num_loss_tokens_to_skip=3,
    )

    trainer = DistilTrainer(
        model=model, ref_model=teacher,
        args=dc, train_dataset=dataset,
        processing_class=processor,
    )
    trainer.train()


if __name__ == "__main__":
    main()
