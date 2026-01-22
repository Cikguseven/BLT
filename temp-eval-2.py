import sys
sys.path.insert(0, "/home/kieron/fyp")

import torch

from pathlib import Path
from typing import Dict, List
import time
from bytelatent.generate import load_consolidated_model_and_tokenizer
from bytelatent.data.patcher import PatcherArgs, PatchingModeEnum
from myte.src.myt5.myt5_tokenizer import MyT5Tokenizer
from tokenizers import Tokenizer

LINES = 1012
EVAL_DIR = Path("/home/kieron/fyp/data_flores-plus_devtest")

LANGS_60 = [
    "eng_Latn",
    "deu_Latn",
    "fra_Latn",
    "ita_Latn",
    "rus_Cyrl",
    "spa_Latn",
    "jpn_Jpan",
    "pol_Latn",
    "por_Latn",
    "vie_Latn",
    "tur_Latn",
    "nld_Latn",
    "ind_Latn",
    "arb_Arab",
    "ces_Latn",
    "pes_Arab",
    "ell_Grek",
    "cmn_Hans",
    "hin_Deva",
    "kor_Hang",
    "tha_Thai",
    "heb_Hebr",
    "ben_Beng",
    "tam_Taml",
    "kat_Geor",
    "mar_Deva",
    "fil_Latn",
    "tel_Telu",
    "nob_Latn",
    "azj_Latn",
    "swe_Latn",
    "ron_Latn",
    "ukr_Cyrl",
    "hun_Latn",
    "dan_Latn",
    "fin_Latn",
    "bul_Cyrl",
    "slk_Latn",
    "cat_Latn",
    "zsm_Latn",
    "urd_Arab",
    "bel_Cyrl",
    "eus_Latn",
    "tgk_Cyrl",
    "sot_Latn",
    "yor_Latn",
    "swh_Latn",
    "ekk_Latn",
    "lvs_Latn",
    "glg_Latn",
    "cym_Latn",
    "als_Latn",
    "mkd_Cyrl",
    "mal_Mlym",
    "mya_Mymr",
    "guj_Gujr",
    "afr_Latn",
    "uzn_Latn",
]

LANGS_30 = [
    "eng_Latn",
    "deu_Latn",
    "fra_Latn",
    "ita_Latn",
    "rus_Cyrl",
    "spa_Latn",
    "jpn_Jpan",
    "pol_Latn",
    "por_Latn",
    "vie_Latn",
    "tur_Latn",
    "nld_Latn",
    "ind_Latn",
    "arb_Arab",
    "ces_Latn",
    "pes_Arab",
    "ell_Grek",
    "cmn_Hans",
    "hin_Deva",
    "kor_Hang",
    "tha_Thai",
    "heb_Hebr",
    "ben_Beng",
    "tam_Taml",
    "kat_Geor",
    "mar_Deva",
    "fil_Latn",
    "tel_Telu",
    "nob_Latn",
    "azj_Latn",
]

SEA_LANGS = [
    "eng_Latn",
    "ind_Latn",
    "fil_Latn",
    "vie_Latn",
    "zsm_Latn",
    "khm_Khmr",
    "lao_Laoo",
    "mya_Mymr",
    "tha_Thai",
    "tam_Taml",
    "cmn_Hans"
]

VALID_LANGS = LANGS_30

# Simple byte tokenizer that segments text into UTF-8 bytes
class ByteTokenizer:
    def __call__(self, lines: List[str], padding: bool = False, add_special_tokens: bool = False):
        masks: List[List[int]] = []
        for line in lines:
            n_bytes = len(line.encode("utf-8"))
            masks.append([1] * n_bytes)
        return {"attention_mask": masks}


def read_lines(fp: Path, max_lines: int) -> List[str]:
    lines: List[str] = []
    with fp.open("r", encoding="utf-8") as f:
        for _ in range(max_lines):
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if line:
                lines.append(line)
    return lines


def count_tokens(tokenizer: MyT5Tokenizer | ByteTokenizer, lines: List[str]) -> int:
    if not lines:
        return 0
    tokens = tokenizer(
        lines,
        padding=False,
        add_special_tokens=False
    )
    return sum(len(sublist) for sublist in tokens["attention_mask"])


def count_parity_tokens(tokenizer: Tokenizer, lines: List[str]) -> int:
    if not lines:
        return 0
    tokens = tokenizer.encode_batch(lines)
    return sum(len(enc.ids) for enc in tokens)


def count_blt_patches(tokenizer, entropy_model_dir, lines: List[str]) -> int:
    patcher_args = PatcherArgs(
        patching_mode=PatchingModeEnum.entropy,
        realtime_patching=True,
        entropy_model_checkpoint_dir=str(entropy_model_dir),
        patching_device="cuda",
        device="cuda",
    )
    patcher = patcher_args.build()

    total_patches = 0

    for prompt in lines:
        token_ids = tokenizer.encode(prompt)
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        tokens_tensor = torch.tensor([token_ids], dtype=torch.long, device="cuda")
        patch_lengths, _ = patcher.patch(tokens_tensor, include_next_token=False)
        total_patches += len(patch_lengths.squeeze(0).tolist())

    return total_patches


# Verified 22 Jan
def gini(values: List[float]) -> float:
    """Compute the Gini coefficient for a list of non-negative values."""
    vals = [v for v in values if v >= 0]
    n = len(vals)
    if n < 2:
        return 0.0
    vals.sort()
    total = sum(vals)
    if total == 0:
        return 0.0
    weighted_sum = 0.0
    for i, v in enumerate(vals, start=1):  # 1-indexed
        weighted_sum += i * v
    return (2 * weighted_sum) / (n * total) - (n + 1) / n


def main() -> None:
    myte_tokenizer = MyT5Tokenizer()
    byte_tokenizer = ByteTokenizer()

    parity_aware_bpe_tokenizer = Tokenizer.from_file("/home/kieron/fyp/parity_aware_bpe/128k_base_30lang_unbalanced_5m/tokenizer.json")

    entropy_model_dir = Path("/home/kieron/fyp/blt/hf-weights/entropy_model")
    checkpoint_path = Path("/home/kieron/fyp/blt/hf-weights/blt_1b")
    _, blt_tokenizer, _ = load_consolidated_model_and_tokenizer(checkpoint_path)

    # Byte-based metrics
    tokens_per_lang_bytes: Dict[str, int] = {}
    avg_tokens_bytes: Dict[str, float] = {}
    avg_parity_bytes: Dict[str, float] = {}

    # Metrics for MYTE
    tokens_per_lang_myte: Dict[str, int] = {}
    avg_tokens_myte: Dict[str, float] = {}
    avg_parity_myte: Dict[str, float] = {}

    # Metrics for Parity-Aware BPE
    tokens_per_lang_pa_bpe: Dict[str, int] = {}
    avg_tokens_pa_bpe: Dict[str, float] = {}
    avg_parity_pa_bpe: Dict[str, float] = {}

    # Metrics for BLT
    tokens_per_lang_blt: Dict[str, int] = {}
    avg_tokens_blt: Dict[str, float] = {}
    avg_parity_blt: Dict[str, float] = {}

    if not EVAL_DIR.is_dir():
        raise FileNotFoundError(f"Directory not found: {EVAL_DIR}")

    output_lines: List[str] = []

    output_lines.append(f"Evaluation started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

    for entry in sorted(EVAL_DIR.iterdir()):
        if not entry.is_file():
            continue
        name = entry.name
        code = name.rsplit(".", 1)[0]
        if code not in VALID_LANGS:
            continue
        print(f"Processing language: {code}")

        lines = read_lines(entry, LINES)
        num_lines = len(lines)

        total_bytes = count_tokens(byte_tokenizer, lines)
        total_myte = count_tokens(myte_tokenizer, lines)
        total_pa_bpe = count_parity_tokens(parity_aware_bpe_tokenizer, lines)
        total_blt = count_blt_patches(blt_tokenizer, entropy_model_dir, lines)

        output_lines.append(f"Processed {code}: MYTE tokens={total_myte}, UTF-8 bytes={total_bytes}, Parity-Aware BPE tokens={total_pa_bpe}, BLT patches={total_blt}")

        output_lines.append(f"Processed {code}: Parity-Aware BPE tokens={total_pa_bpe}")

        if total_bytes > 0:
            tokens_per_lang_bytes[code] = total_bytes
            avg_tokens_bytes[code] = total_bytes / num_lines
        if total_myte > 0:
            tokens_per_lang_myte[code] = total_myte
            avg_tokens_myte[code] = total_myte / num_lines
        if total_pa_bpe > 0:
            tokens_per_lang_pa_bpe[code] = total_pa_bpe
            avg_tokens_pa_bpe[code] = total_pa_bpe / num_lines
        if total_blt > 0:
            tokens_per_lang_blt[code] = total_blt
            avg_tokens_blt[code] = total_blt / num_lines

    # Use correct English key
    eng_tokens_bytes = tokens_per_lang_bytes.get("eng_Latn") or tokens_per_lang_bytes.get("eng")
    eng_tokens_myte = tokens_per_lang_myte.get("eng_Latn") or tokens_per_lang_myte.get("eng")
    eng_tokens_pa_bpe = tokens_per_lang_pa_bpe.get("eng_Latn") or tokens_per_lang_pa_bpe.get("eng")
    eng_tokens_blt = tokens_per_lang_blt.get("eng_Latn") or tokens_per_lang_blt.get("eng")

    for lang in sorted(set(
        list(avg_tokens_bytes.keys()) +
        list(avg_tokens_myte.keys()) +
        list(avg_tokens_pa_bpe.keys()) +
        list(avg_tokens_blt.keys())
    )):
        parts = [f"{lang}:"]
        if lang in avg_tokens_bytes:
            if eng_tokens_bytes:
                parity_bytes = tokens_per_lang_bytes[lang] / eng_tokens_bytes
                avg_parity_bytes[lang] = parity_bytes
                lines_per_token_bytes = 1 / avg_tokens_bytes[lang] if avg_tokens_bytes[lang] > 0 else 0.0
                parts.append(f"UTF-8 parity: {parity_bytes:.2f}")
                parts.append(f"UTF-8 average bytes per line: {avg_tokens_bytes[lang]:.2f}")
                parts.append(f"UTF-8 compression rate: {lines_per_token_bytes:.4f}")
        if lang in avg_tokens_myte:
            if eng_tokens_myte:
                parity_myte = tokens_per_lang_myte[lang] / eng_tokens_myte
                avg_parity_myte[lang] = parity_myte
                lines_per_token_myte = 1 / avg_tokens_myte[lang] if avg_tokens_myte[lang] > 0 else 0.0
                parts.append(f"MYTE parity: {parity_myte:.2f}")
                parts.append(f"MYTE average tokens per line: {avg_tokens_myte[lang]:.2f}")
                parts.append(f"MYTE compression rate: {lines_per_token_myte:.4f}")
        if lang in avg_tokens_pa_bpe:
            if eng_tokens_pa_bpe:
                parity_pa_bpe = tokens_per_lang_pa_bpe[lang] / eng_tokens_pa_bpe
                avg_parity_pa_bpe[lang] = parity_pa_bpe
                lines_per_token_pa_bpe = 1 / avg_tokens_pa_bpe[lang] if avg_tokens_pa_bpe[lang] > 0 else 0.0
                parts.append(f"Parity-Aware BPE parity: {parity_pa_bpe:.2f}")
                parts.append(f"Parity-Aware BPE average tokens per line: {avg_tokens_pa_bpe[lang]:.2f}")
            parts.append(f"Parity-Aware BPE compression rate: {lines_per_token_pa_bpe:.2f}")
        if lang in avg_tokens_blt:
            if eng_tokens_blt:
                parity_blt = tokens_per_lang_blt[lang] / eng_tokens_blt
                avg_parity_blt[lang] = parity_blt
                lines_per_token_blt = 1 / avg_tokens_blt[lang] if avg_tokens_blt[lang] > 0 else 0.0
                parts.append(f"BLT parity: {parity_blt:.2f}")
                parts.append(f"BLT average patches per line: {avg_tokens_blt[lang]:.2f}")
            parts.append(f"BLT compression rate: {lines_per_token_blt:.2f}")
        output_lines.append(", ".join(parts))

    # Gini coefficient across languages
    gini_bytes = gini(list(avg_tokens_bytes.values()))
    gini_myte = gini(list(avg_tokens_myte.values()))
    gini_SEA_parity = gini(list(avg_tokens_pa_bpe.values()))
    gini_blt = gini(list(avg_tokens_blt.values()))

    output_lines.append("=== Gini Coefficients (across languages) ===")
    output_lines.append(
        f"MYTE: {gini_myte:.3f}, UTF-8: {gini_bytes:.3f}, Parity-Aware BPE: {gini_SEA_parity:.3f}, BLT: {gini_blt:.3f}"
    )

    langs_with_bytes = len(tokens_per_lang_bytes)
    langs_with_myte = len(tokens_per_lang_myte)
    langs_with_pa_bpe = len(tokens_per_lang_pa_bpe)
    langs_with_blt = len(tokens_per_lang_blt)

    # Global average tokens per line across languages
    combined_tokens_per_line_bytes = (sum(avg_tokens_bytes.values()) / langs_with_bytes) if langs_with_bytes > 0 else 0.0
    combined_tokens_per_line_myte = (sum(avg_tokens_myte.values()) / langs_with_myte) if langs_with_myte > 0 else 0.0
    combined_tokens_per_line_pa_bpe = (sum(avg_tokens_pa_bpe.values()) / langs_with_pa_bpe) if langs_with_pa_bpe > 0 else 0.0
    combined_tokens_per_line_blt = (sum(avg_tokens_blt.values()) / langs_with_blt) if langs_with_blt > 0 else 0.0

    # Global compression rate across languages
    combined_lines_per_token_bytes = 1 / combined_tokens_per_line_bytes if combined_tokens_per_line_bytes > 0 else 0.0
    combined_lines_per_token_myte = 1 / combined_tokens_per_line_myte if combined_tokens_per_line_myte > 0 else 0.0
    combined_lines_per_token_pa_bpe = 1 / combined_tokens_per_line_pa_bpe if combined_tokens_per_line_pa_bpe > 0 else 0.0
    combined_lines_per_token_blt = 1 / combined_tokens_per_line_blt if combined_tokens_per_line_blt > 0 else 0.0

    # Average parity across languages
    combined_parity_bytes = sum(avg_parity_bytes.values()) / langs_with_bytes if (langs_with_bytes) else 0.0
    combined_parity_myte = sum(avg_parity_myte.values()) / langs_with_myte if (langs_with_myte) else 0.0
    combined_parity_pa_bpe = sum(avg_parity_pa_bpe.values()) / langs_with_pa_bpe if (langs_with_pa_bpe) else 0.0
    combined_parity_blt = sum(avg_parity_blt.values()) / langs_with_blt if (langs_with_blt) else 0.0

    output_lines.append("=== Average tokens per Line (lower is better) ===")
    output_lines.append(f"UTF-8: {combined_tokens_per_line_bytes:.6f}")
    output_lines.append(f"MYTE: {combined_tokens_per_line_myte:.6f}")
    output_lines.append(f"Parity-Aware BPE: {combined_tokens_per_line_pa_bpe:.6f}")
    output_lines.append(f"BLT: {combined_tokens_per_line_blt:.6f}")

    output_lines.append("=== Compression Rates across all languages (higher is better) ===")
    output_lines.append(f"UTF-8: {combined_lines_per_token_bytes:.6f}")
    output_lines.append(f"MYTE: {combined_lines_per_token_myte:.6f}")
    output_lines.append(f"Parity-Aware BPE: {combined_lines_per_token_pa_bpe:.6f}")
    output_lines.append(f"BLT: {combined_lines_per_token_blt:.6f}")

    output_lines.append("=== Average Parity vs English (lower is better) ===")
    output_lines.append(f"UTF-8: {combined_parity_bytes:.6f}")
    output_lines.append(f"MYTE: {combined_parity_myte:.6f}")
    output_lines.append(f"Parity-Aware BPE: {combined_parity_pa_bpe:.6f}")
    output_lines.append(f"BLT: {combined_parity_blt:.6f}")

    output_lines.append("=== Worst-case Parity vs English (lower is better) ===")
    output_lines.append(f"UTF-8: {max(avg_parity_bytes.values()) if eng_tokens_bytes else "English missing"}")
    output_lines.append(f"MYTE: {max(avg_parity_myte.values()) if eng_tokens_myte else "English missing"}")
    output_lines.append(f"Parity-Aware BPE: {max(avg_parity_pa_bpe.values()) if eng_tokens_pa_bpe else "English missing"}")
    output_lines.append(f"BLT: {max(avg_parity_blt.values()) if eng_tokens_blt else "English missing"}")

    # Write output to file instead of printing
    timestamp = time.strftime("%b%d-%H%M", time.localtime())
    output_path = Path(f"/home/kieron/fyp/output_tokenizer_eval_{timestamp}.log")
    with open(output_path, "w", encoding="utf-8") as f:
        for line in output_lines:
            f.write(line + "\n\n")

if __name__ == "__main__":
    main()