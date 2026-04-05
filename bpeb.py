import os
from pathlib import Path
import sys
sys.path.insert(0, "/localhome/kieron")

import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import math

LN2 = math.log(2)
MAX_LENGTH   = 4096
STRIDE       = MAX_LENGTH
MAX_BYTES    = 4096
STRIDE_BYTES = MAX_BYTES

FLORES_DIR = Path("/localhome/kieron/fyp/data/flores-plus_dev_devtest")
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

HF_MODEL_PATHS = {
    "MYTE-269B"           : "/localhome/kieron/fyp/models/myte-269b",
    "PARITY-AWARE-BPE-82B": "/localhome/kieron/fyp/models/parity-aware-bpe-82b",
    "BYTE-LEVEL-BPE-72B"  : "/localhome/kieron/fyp/models/byte-level-bpe-72b",
}

BLT_MODEL_CONFIGS = {
    "BLT-42B": {
        "checkpoint_path"  : "/localhome/kieron/fyp/models/blt-42b/model",
        "entropy_model_dir": "/localhome/kieron/fyp/models/blt-42b/entropy",
    },
}

SEA_11 = [
    "eng_Latn", "ind_Latn", "vie_Latn", "zsm_Latn", "khm_Khmr",
    "lao_Laoo", "mya_Mymr", "tha_Thai", "tam_Taml", "cmn_Hans",
]


# ── Pure functions — safe to define at module level ───────────────────────────

def load_flores(lang_code: str) -> list[str]:
    fpath = FLORES_DIR / f"{lang_code}.devtest"
    if not fpath.exists():
        raise FileNotFoundError(f"FLORES file not found: {fpath}")
    with open(fpath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


@torch.no_grad()
def compute_bpeb_sentence(sentence, english_reference, model, tokenizer):
    n_english_bytes = len(english_reference.encode("utf-8")) + 1
    input_ids = tokenizer(
        sentence, return_tensors="pt", add_special_tokens=True
    ).input_ids.to(DEVICE)
    eos_id    = torch.tensor([[tokenizer.eos_token_id]], device=DEVICE)
    input_ids = torch.cat([input_ids, eos_id], dim=1)
    T = input_ids.size(1)
    total_nll_nats = 0.0
    prev_end = 0
    for begin in range(0, T, STRIDE):
        end        = min(begin + MAX_LENGTH, T)
        target_len = end - prev_end
        chunk      = input_ids[:, begin:end]
        labels     = chunk.clone()
        labels[:, : chunk.size(1) - target_len] = -100
        loss       = model(chunk, labels=labels).loss
        total_nll_nats += loss.item() * (target_len - 1)
        prev_end = end
        if end == T:
            break
    total_nll_bits = total_nll_nats / LN2
    return {
        "bpeb": total_nll_bits / n_english_bytes,
        "nll_bits": total_nll_bits,
        "n_tokens": T,
        "n_english_bytes": n_english_bytes,
    }


@torch.inference_mode()
def compute_bpeb_sentence_blt(sentence, english_reference, model, tokenizer, patcher):
    n_english_bytes = len(english_reference.encode("utf-8")) + 1
    byte_ids = tokenizer.encode(sentence, add_eos=False) + [tokenizer.eos_id]
    tokens   = torch.tensor(byte_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
    T        = tokens.size(1)
    total_nll_nats = 0.0
    prev_end = 0
    for begin in range(0, T, STRIDE_BYTES):
        end        = min(begin + MAX_BYTES, T)
        target_len = end - prev_end
        chunk      = tokens[:, begin:end]
        patch_lengths, _ = patcher.patch(chunk, include_next_token=False)
        logits    = model(chunk, patch_lengths=patch_lengths)
        ctx_offset = chunk.size(1) - target_len
        pred_logits = logits[:, ctx_offset:-1, :]
        targets     = chunk[:, ctx_offset + 1:]
        if target_len > 1:
            total_nll_nats += F.cross_entropy(
                pred_logits.reshape(-1, pred_logits.size(-1)),
                targets.reshape(-1),
                reduction="sum",
            ).item()
        prev_end = end
        if end == T:
            break
    total_nll_bits = total_nll_nats / LN2
    return {
        "bpeb": total_nll_bits / n_english_bytes,
        "nll_bits": total_nll_bits,
        "n_tokens": T,
        "n_english_bytes": n_english_bytes,
    }


def aggregate(lang_results):
    total_nll_bits  = sum(r["nll_bits"]        for r in lang_results)
    total_eng_bytes = sum(r["n_english_bytes"] for r in lang_results)
    return {
        "corpus_bpeb"    : total_nll_bits / total_eng_bytes,
        "mean_bpeb"      : sum(r["bpeb"]     for r in lang_results) / len(lang_results),
        "mean_tokens"    : sum(r["n_tokens"] for r in lang_results) / len(lang_results),
        "total_sentences": len(lang_results),
    }


# ── ALL side-effectful / process-spawning code lives here ─────────────────────
# ↓↓↓ THIS GUARD IS THE FIX ↓↓↓
if __name__ == "__main__":

    # 1. Load FLORES
    print("Loading FLORES-200 devtest sentences...")
    flores_data = {lang: load_flores(lang) for lang in SEA_11}
    n_sentences = len(flores_data["eng_Latn"])
    for lang, sents in flores_data.items():
        assert len(sents) == n_sentences
    print(f"Loaded {n_sentences} sentences × {len(SEA_11)} languages.\n")
    english_sentences = flores_data["eng_Latn"]

    os.makedirs("output", exist_ok=True)
    all_summary_rows  = []
    all_per_sent_rows = []

    # 2. HuggingFace models
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

    for model_name, model_path in HF_MODEL_PATHS.items():
        print(f"\n{'='*65}\nLoading {model_name} ...\n{'='*65}")
        cfg       = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
        model     = AutoModelForCausalLM.from_pretrained(
            model_path, config=cfg, torch_dtype="auto",
            device_map="auto", local_files_only=True, trust_remote_code=True,
        )
        model.eval()

        for lang in SEA_11:
            lang_results = []
            for sent, eng_ref in tqdm(
                zip(flores_data[lang], english_sentences),
                total=n_sentences, desc=f"{model_name}/{lang}", unit="sent",
            ):
                res = compute_bpeb_sentence(sent, eng_ref, model, tokenizer)
                lang_results.append(res)
                all_per_sent_rows.append({
                    "model": model_name, "language": lang,
                    "sentence_idx": len(lang_results) - 1,
                    **{k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()},
                })
            agg = aggregate(lang_results)
            all_summary_rows.append({"model": model_name, "language": lang,
                "corpus_bpeb": round(agg["corpus_bpeb"], 4),
                "mean_bpeb"  : round(agg["mean_bpeb"],   4),
                "mean_tokens": round(agg["mean_tokens"],  2),
                "total_sentences": agg["total_sentences"],
            })

        del model, tokenizer
        torch.cuda.empty_cache()

    # 3. BLT models  ← distributed setup is now INSIDE the guard
    from bytelatent.distributed import DistributedArgs, setup_torch_distributed
    from bytelatent.generate import load_consolidated_model_and_tokenizer
    from bytelatent.model.blt import ByteLatentTransformer
    from bytelatent.blt_tokenizers.blt_tokenizer import BltTokenizer

    distributed_args = DistributedArgs()
    distributed_args.configure_world()
    if not torch.distributed.is_initialized():
        setup_torch_distributed(distributed_args)

    for model_name, blt_cfg in BLT_MODEL_CONFIGS.items():
        print(f"\n{'='*65}\nLoading BLT model: {model_name} ...\n{'='*65}")
        checkpoint_path   = Path(blt_cfg["checkpoint_path"])
        entropy_model_dir = Path(blt_cfg["entropy_model_dir"])

        blt_model, blt_tokenizer, train_cfg = load_consolidated_model_and_tokenizer(checkpoint_path)
        assert isinstance(blt_model,     ByteLatentTransformer)
        assert isinstance(blt_tokenizer, BltTokenizer)

        patcher_args = train_cfg.data.patcher_args.model_copy(deep=True)
        patcher_args.realtime_patching            = True
        patcher_args.entropy_model_checkpoint_dir = str(entropy_model_dir)
        patcher = patcher_args.build()
        blt_model.eval()

        for lang in SEA_11:
            lang_results = []
            for sent, eng_ref in tqdm(
                zip(flores_data[lang], english_sentences),
                total=n_sentences, desc=f"{model_name}/{lang}", unit="sent",
            ):
                res = compute_bpeb_sentence_blt(sent, eng_ref, blt_model, blt_tokenizer, patcher)
                lang_results.append(res)
                all_per_sent_rows.append({
                    "model": model_name, "language": lang,
                    "sentence_idx": len(lang_results) - 1,
                    **{k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()},
                })
            agg = aggregate(lang_results)
            all_summary_rows.append({"model": model_name, "language": lang,
                "corpus_bpeb": round(agg["corpus_bpeb"], 4),
                "mean_bpeb"  : round(agg["mean_bpeb"],   4),
                "mean_tokens": round(agg["mean_tokens"],  2),
                "total_sentences": agg["total_sentences"],
            })

        del blt_model, blt_tokenizer, patcher
        torch.cuda.empty_cache()

    # 4. Save
    df_summary  = pd.DataFrame(all_summary_rows)
    df_per_sent = pd.DataFrame(all_per_sent_rows)
    print("\n" + "="*65)
    print(df_summary.to_string(index=False))
    os.makedirs("/localhome/kieron/fyp/bpeb", exist_ok=True)
    df_summary.to_csv("/localhome/kieron/fyp/bpeb/results.csv",       index=False)
    print("\nSaved to /localhome/kieron/fyp/bpeb/")