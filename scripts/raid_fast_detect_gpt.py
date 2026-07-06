import argparse
import gc
import json
import os
import sys
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from fast_detect_gpt import get_sampling_discrepancy_analytic
from metrics import get_precision_recall_metrics, get_roc_metrics


REPO_ROOT = Path(__file__).resolve().parents[2]

ENCODING_MODELS = {
    "encoding_mistral": "mistralai/Mistral-7B-v0.1",
    "encoding_mistral-instruct": "mistralai/Mistral-7B-Instruct-v0.2",
}

DEFAULT_DOMAINS = [
    "abstracts",
    "news",
    "recipes",
    "reddit",
    "reviews",
    "wiki",
    "books",
    "poetry",
]

GENERATOR_FOLDERS_BY_REPETITION = {
    "no": {
        "chatgpt": ("encoding_mistral-instruct", "chatgpt"),
        "cohere-chat": ("encoding_mistral-instruct", "cohere-chat"),
        "gpt4": ("encoding_mistral-instruct", "gpt4"),
        "cohere": ("encoding_mistral", "cohere"),
        "gpt3": ("encoding_mistral", "gpt3"),
    },
    "yes": {
        "gpt2": ("encoding_mistral", "gpt2"),
        "mistral": ("encoding_mistral", "mistral-base"),
        "mpt": ("encoding_mistral", "mpt"),
        "llama-chat": ("encoding_mistral", "chat_models/llama-chat"),
        "llama-chat-instruct": ("encoding_mistral-instruct", "chat_models/llama-chat"),
        "mistral-chat": ("encoding_mistral-instruct", "chat_models/mistral-chat"),
        "mpt-chat": ("encoding_mistral-instruct", "chat_models/mpt-chat"),
    },
}

DATASET_MODEL_NAMES = {
    "llama-chat-instruct": "llama-chat",
}

RAID_TRAIN_URL = "https://huggingface.co/datasets/liamdugan/raid/resolve/main/train.csv"
RAID_TRAIN_CACHE_PATH = Path("/media/pinas/cache/huggingface/datasets/downloads/0e8ed8a98b8a4bec7eaf1300ce9be88fd131e33751cc5ab2183c37ef93886e6d")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Fast-DetectGPT on paired human/LLM RAID examples."
    )
    parser.add_argument(
        "--encoding-root",
        choices=sorted(ENCODING_MODELS),
        required=True,
        help="Top-level output/model family, matching the existing result folders.",
    )
    parser.add_argument(
        "--generators",
        nargs="+",
        default=None,
        help="RAID model names to evaluate. Defaults to generators available for --encoding-root.",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=DEFAULT_DOMAINS,
        help="RAID domains to evaluate.",
    )
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument(
        "--repetition-penalty",
        choices=["yes", "no"],
        default="no",
        help="RAID repetition_penalty value to evaluate and output under rep_penalty_<value>.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--output-baseline-name",
        type=str,
        default="fast_detect_gpt",
        help="Folder name under full_dataset_exp/baselines.",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Load the model in fp16 instead of 4-bit NF4.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute domain files that already exist.",
    )
    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(model_name, device, cache_dir, use_4bit=True):
    tokenizer_kwargs = {"trust_remote_code": True, "padding_side": "right"}
    model_kwargs = {"trust_remote_code": True}
    if cache_dir:
        tokenizer_kwargs["cache_dir"] = cache_dir
        model_kwargs["cache_dir"] = cache_dir

    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model_kwargs["device_map"] = {"": device}
    else:
        model_kwargs["torch_dtype"] = torch.float16 if device.startswith("cuda") else torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if not use_4bit:
        model = model.to(device)
    model.eval()
    return model, tokenizer


def paired_raid_by_domain(generator, domains, cache_dir, repetition_penalty):
    train_data_file = str(RAID_TRAIN_CACHE_PATH if RAID_TRAIN_CACHE_PATH.exists() else RAID_TRAIN_URL)
    print(f"Loading RAID train data from {train_data_file}")
    dataset = datasets.load_dataset(
        "csv", data_files={"train": train_data_file}, split="train", cache_dir=cache_dir
    )
    filtered = dataset.filter(
        lambda x: (
            x["domain"] in domains
            and x["model"] in ["human", DATASET_MODEL_NAMES.get(generator, generator)]
            and x["attack"] == "none"
            and (
                x["model"] == "human"
                or (
                    x["decoding"] == "sampling"
                    and (
                        repetition_penalty == "no"
                        or x["repetition_penalty"] == repetition_penalty
                    )
                )
            )
        )
    )

    by_domain = {}
    for domain in domains:
        domain_dataset = filtered.filter(lambda x: x["domain"] == domain)
        df = domain_dataset.to_pandas().drop_duplicates(subset=["source_id", "model"])
        if df.empty:
            by_domain[domain] = []
            continue

        paired_df = (
            df.pivot(index="source_id", columns="model", values="generation")
            .reset_index()
            .dropna(subset=["human", DATASET_MODEL_NAMES.get(generator, generator)])
        )
        by_domain[domain] = [
            {"human": row["human"], "llm": row[DATASET_MODEL_NAMES.get(generator, generator)]}
            for _, row in paired_df.iterrows()
        ]
    return by_domain


def fast_detect_score(text, model, tokenizer, device, max_length):
    tokenized = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        return_token_type_ids=False,
    ).to(device)

    if tokenized.input_ids.size(1) < 2:
        return np.nan

    labels = tokenized.input_ids[:, 1:]
    with torch.no_grad():
        logits = model(**tokenized).logits[:, :-1]
        return get_sampling_discrepancy_analytic(logits, logits, labels)


def score_domain(data, domain, output_dir, model, tokenizer, args):
    human_path = output_dir / f"human_{domain}.pt"
    llm_path = output_dir / f"llm_{domain}.pt"
    meta_path = output_dir / f"metadata_{domain}.json"

    if human_path.exists() and llm_path.exists() and not args.overwrite:
        print(f"Skipping {domain}: existing files found in {output_dir}")
        return

    human_scores = []
    llm_scores = []

    for start in range(0, len(data), args.batch_size):
        batch = data[start:start + args.batch_size]
        for pair in tqdm(batch, desc=f"{domain} {start}-{start + len(batch)}"):
            human_scores.append(
                fast_detect_score(pair["human"], model, tokenizer, args.device, args.max_length)
            )
            llm_scores.append(
                fast_detect_score(pair["llm"], model, tokenizer, args.device, args.max_length)
            )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(human_scores, human_path)
    torch.save(llm_scores, llm_path)

    valid = np.isfinite(human_scores) & np.isfinite(llm_scores)
    metadata = {
        "domain": domain,
        "n_pairs": len(data),
        "n_valid_pairs": int(valid.sum()),
        "score": "fast_detect_gpt_sampling_discrepancy_analytic",
        "max_length": args.max_length,
    }

    if valid.any():
        human_valid = np.asarray(human_scores, dtype=float)[valid].tolist()
        llm_valid = np.asarray(llm_scores, dtype=float)[valid].tolist()
        fpr, tpr, roc_auc = get_roc_metrics(human_valid, llm_valid)
        precision, recall, pr_auc = get_precision_recall_metrics(human_valid, llm_valid)
        metadata["metrics"] = {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "human_mean": float(np.mean(human_valid)),
            "llm_mean": float(np.mean(llm_valid)),
        }
        metadata["roc"] = {"fpr": fpr, "tpr": tpr}
        metadata["pr"] = {"precision": precision, "recall": recall}

    with meta_path.open("w") as fout:
        json.dump(metadata, fout, indent=2)

    print(f"Saved Fast-DetectGPT scores for {domain} to {output_dir}")


def generator_folders_for(repetition_penalty):
    return GENERATOR_FOLDERS_BY_REPETITION[repetition_penalty]


def generators_for_encoding_root(encoding_root, requested, repetition_penalty):
    if requested:
        return requested
    return [
        generator
        for generator, (root, _folder) in generator_folders_for(repetition_penalty).items()
        if root == encoding_root
    ]


def output_dir_for(args, generator):
    expected_root, folder = generator_folders_for(args.repetition_penalty).get(
        generator, (args.encoding_root, generator)
    )
    if expected_root != args.encoding_root:
        raise ValueError(
            f"Generator {generator!r} belongs to {expected_root}, not {args.encoding_root}."
        )
    return (
        REPO_ROOT
        / args.encoding_root
        / f"rep_penalty_{args.repetition_penalty}"
        / folder
        / "full_dataset_exp"
        / "baselines"
        / args.output_baseline_name
    )


def main():
    args = parse_args()
    if args.device.startswith("cuda"):
        assert torch.cuda.is_available(), "CUDA is not available"

    set_seed(args.seed)
    model_name = ENCODING_MODELS[args.encoding_root]
    model, tokenizer = load_model_and_tokenizer(
        model_name,
        args.device,
        args.cache_dir,
        use_4bit=not args.no_4bit,
    )

    for generator in generators_for_encoding_root(
        args.encoding_root, args.generators, args.repetition_penalty
    ):
        output_dir = output_dir_for(args, generator)
        print(f"\nRunning {generator} with {model_name}")
        print(f"Output: {output_dir}")

        domain_data = paired_raid_by_domain(
            generator, args.domains, args.cache_dir, args.repetition_penalty
        )
        for domain in args.domains:
            data = domain_data[domain]
            print(f"{domain}: {len(data)} paired examples")
            if not data:
                continue
            score_domain(data, domain, output_dir, model, tokenizer, args)


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT / "fast-detect-gpt" / "scripts"))
    main()
