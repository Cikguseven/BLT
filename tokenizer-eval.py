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
DIR_PATH = Path("/home/kieron/fyp/data_flores-plus_devtest")

VALID_LANGS = [
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

def read_lines(fp: Path, max_lines: int) -> List[str]:
    lines: List[str] = []
    with fp.open("r", encoding="utf-8") as f:
        for _ in range(max_lines):
            line = f.readline()
            line = line.strip()
            if not line:
                break
            if line:
                lines.append(line)
    return lines


def count_tokens(tokenizer: MyT5Tokenizer, lines: List[str]) -> int:
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


# Simple byte tokenizer that segments text into UTF-8 bytes
class ByteTokenizer:
    def __call__(self, lines: List[str], padding: bool = False, add_special_tokens: bool = False):
        masks: List[List[int]] = []
        for line in lines:
            n_bytes = len(line.encode("utf-8"))
            masks.append([1] * n_bytes)
        return {"attention_mask": masks}


def main() -> None:
    myte_tokenizer = MyT5Tokenizer()
    byte_tokenizer = ByteTokenizer()

    parity_aware_bpe_tokenizer = Tokenizer.from_file("/home/kieron/fyp/parity_aware_bpe/45k_parity-aware_SEA_1m/tokenizer.json")
    byte_level_bpe_tokenizer = Tokenizer.from_file("/home/kieron/fyp/parity_aware_bpe/45k_byte-level_SEA_1m/tokenizer.json")
    char_level_bpe_tokenizer = Tokenizer.from_file("/home/kieron/fyp/parity_aware_bpe/45k_char-level_SEA_1m/tokenizer.json")

    entropy_model_dir = Path("/home/kieron/fyp/blt/hf-weights/entropy_model")
    checkpoint_path = Path("/home/kieron/fyp/blt/hf-weights/blt_1b")
    _, blt_tokenizer, _ = load_consolidated_model_and_tokenizer(checkpoint_path)

    # Metrics
    token_totals_myte: Dict[str, int] = {}
    avg_tokens_myte: Dict[str, float] = {}

    # Byte-based metrics
    token_totals_bytes: Dict[str, int] = {}
    avg_tokens_bytes: Dict[str, float] = {}

    # Parity-based metrics
    SEA_token_totals_parity: Dict[str, int] = {}
    SEA_avg_tokens_parity: Dict[str, float] = {}

    byte_level_SEA_token_totals_parity: Dict[str, int] = {}
    byte_level_SEA_avg_tokens_parity: Dict[str, float] = {}

    char_level_SEA_token_totals_parity: Dict[str, int] = {}
    char_level_SEA_avg_tokens_parity: Dict[str, float] = {}

    # BLT-based metrics
    blt_token_patches: Dict[str, int] = {}
    blt_avg_patches: Dict[str, float] = {}

    if not DIR_PATH.is_dir():
        raise FileNotFoundError(f"Directory not found: {DIR_PATH}")

    output_lines: List[str] = []

    output_lines.append(f"Evaluation started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

    for entry in sorted(DIR_PATH.iterdir()):
        if not entry.is_file():
            continue
        name = entry.name
        code = name.rsplit(".", 1)[0]

        if code not in VALID_LANGS:
            continue
        print(f"Processing language: {code}")
        lines = read_lines(entry, LINES)

        total_tokens = count_tokens(myte_tokenizer, lines)
        total_bytes = count_tokens(byte_tokenizer, lines)
        total_SEA_parity = count_parity_tokens(parity_aware_bpe_tokenizer, lines)
        total_byte_level_SEA_parity = count_parity_tokens(byte_level_bpe_tokenizer, lines)
        total_char_level_SEA_parity = count_parity_tokens(char_level_bpe_tokenizer, lines)
        total_blt_patches = count_blt_patches(blt_tokenizer, entropy_model_dir, lines)

        output_lines.append(f"Processed {code}: MYTE tokens={total_tokens}, UTF-8 bytes={total_bytes}, Parity-Aware BPE tokens={total_SEA_parity}, Byte-Level BPE tokens={total_byte_level_SEA_parity}, Char-Level BPE tokens={total_char_level_SEA_parity}, BLT patches={total_blt_patches}")

        if total_tokens == 0 and total_bytes == 0:
            continue

        if total_tokens > 0:
            token_totals_myte[code] = total_tokens
            avg_tokens_myte[code] = total_tokens / LINES
        if total_bytes > 0:
            token_totals_bytes[code] = total_bytes
            avg_tokens_bytes[code] = total_bytes / LINES
        if total_SEA_parity > 0:
            SEA_token_totals_parity[code] = total_SEA_parity
            SEA_avg_tokens_parity[code] = total_SEA_parity / LINES
        if total_byte_level_SEA_parity > 0:
            byte_level_SEA_token_totals_parity[code] = total_byte_level_SEA_parity
            byte_level_SEA_avg_tokens_parity[code] = total_byte_level_SEA_parity / LINES
        if total_char_level_SEA_parity > 0:
            char_level_SEA_token_totals_parity[code] = total_char_level_SEA_parity
            char_level_SEA_avg_tokens_parity[code] = total_char_level_SEA_parity / LINES
        if total_blt_patches > 0:
            blt_token_patches[code] = total_blt_patches
            blt_avg_patches[code] = total_blt_patches / LINES

    # Use correct English key
    eng_tokens = token_totals_myte.get("eng_Latn") or token_totals_myte.get("eng")
    eng_bytes = token_totals_bytes.get("eng_Latn") or token_totals_bytes.get("eng")
    eng_SEA_parity = SEA_token_totals_parity.get("eng_Latn") or SEA_token_totals_parity.get("eng")
    eng_byte_level_SEA_parity = byte_level_SEA_token_totals_parity.get("eng_Latn") or byte_level_SEA_token_totals_parity.get("eng")
    eng_char_level_SEA_parity = char_level_SEA_token_totals_parity.get("eng_Latn") or char_level_SEA_token_totals_parity.get("eng")
    eng_blt_patches = blt_token_patches.get("eng_Latn") or blt_token_patches.get("eng")

    for lang in sorted(set(list(avg_tokens_myte.keys()) + list(avg_tokens_bytes.keys()) + list(SEA_avg_tokens_parity.keys()) + list(byte_level_SEA_avg_tokens_parity.keys()) + list(char_level_SEA_avg_tokens_parity.keys()) + list(blt_avg_patches.keys()))):
        parts = [f"{lang}:"]
        if lang in avg_tokens_bytes:
            if eng_bytes:
                parity_b = token_totals_bytes[lang] / eng_bytes
                avg_bytes_per_line = avg_tokens_bytes[lang]
                avg_lines_per_byte = LINES / token_totals_bytes[lang] if token_totals_bytes[lang] > 0 else 0.0
                parts.append(f"UTF-8 parity: {parity_b:.2f}")
                parts.append(f"UTF-8 average bytes per line: {avg_bytes_per_line:.2f}")
                parts.append(f"UTF-8 compression rate: {avg_lines_per_byte:.4f}")
        if lang in avg_tokens_myte:
            if eng_tokens:
                parity = token_totals_myte[lang] / eng_tokens
                avg_token_len = avg_tokens_myte[lang]
                avg_lines_per_token = LINES / token_totals_myte[lang] if token_totals_myte[lang] > 0 else 0.0
                parts.append(f"MYTE parity: {parity:.2f}")
                parts.append(f"MYTE average tokens per line: {avg_token_len:.2f}")
                parts.append(f"MYTE compression rate: {avg_lines_per_token:.4f}")
        if lang in SEA_avg_tokens_parity:
            fert_sp = SEA_avg_tokens_parity[lang]
            if eng_SEA_parity:
                parity_sp = SEA_token_totals_parity[lang] / eng_SEA_parity
                avg_SEA_parity_token_len = SEA_avg_tokens_parity[lang]
                parts.append(f"Parity-Aware BPE parity: {parity_sp:.2f}")
                parts.append(f"Parity-Aware BPE average tokens per line: {avg_SEA_parity_token_len:.2f}")
            parts.append(f"Parity-Aware BPE compression rate: {fert_sp:.2f}")
        if lang in byte_level_SEA_avg_tokens_parity:
            fert_bsp = byte_level_SEA_avg_tokens_parity[lang]
            if eng_byte_level_SEA_parity:
                parity_bsp = byte_level_SEA_token_totals_parity[lang] / eng_byte_level_SEA_parity
                avg_byte_level_SEA_parity_token_len = byte_level_SEA_avg_tokens_parity[lang]
                parts.append(f"Byte-Level BPE parity: {parity_bsp:.2f}")
                parts.append(f"Byte-Level BPE average tokens per line: {avg_byte_level_SEA_parity_token_len:.2f}")
            parts.append(f"Byte-Level BPE compression rate: {fert_bsp:.2f}")
        if lang in char_level_SEA_avg_tokens_parity:
            fert_csp = char_level_SEA_avg_tokens_parity[lang]
            if eng_char_level_SEA_parity:
                parity_csp = char_level_SEA_token_totals_parity[lang] / eng_char_level_SEA_parity
                avg_char_level_SEA_parity_token_len = char_level_SEA_avg_tokens_parity[lang]
                parts.append(f"Char-Level BPE parity: {parity_csp:.2f}")
                parts.append(f"Char-Level BPE average tokens per line: {avg_char_level_SEA_parity_token_len:.2f}")
            parts.append(f"Char-Level BPE compression rate: {fert_csp:.2f}")
        if lang in blt_avg_patches:
            fert_blt = blt_avg_patches[lang]
            if eng_blt_patches:
                parity_blt = blt_token_patches[lang] / eng_blt_patches
                avg_blt_patches_per_line = blt_avg_patches[lang]
                parts.append(f"BLT parity: {parity_blt:.2f}")
                parts.append(f"BLT average patches per line: {avg_blt_patches_per_line:.2f}")
            parts.append(f"BLT compression rate: {fert_blt:.2f}")
        output_lines.append(", ".join(parts))

    # Gini coefficient summaries across languages
    g_total_myte = gini(list(avg_tokens_myte.values()))
    g_total_bytes = gini(list(avg_tokens_bytes.values()))
    g_total_SEA_parity = gini(list(SEA_avg_tokens_parity.values()))
    g_total_byte_level_SEA_parity = gini(list(byte_level_SEA_avg_tokens_parity.values()))
    g_total_char_level_SEA_parity = gini(list(char_level_SEA_avg_tokens_parity.values()))
    g_total_blt = gini(list(blt_avg_patches.values()))

    output_lines.append("=== Gini Coefficients (across languages) ===")
    output_lines.append(
        f"Token Totals Gini -> MYTE: {g_total_myte:.3f}, UTF-8: {g_total_bytes:.3f}, Parity-Aware BPE: {g_total_SEA_parity:.3f}, Byte-Level BPE: {g_total_byte_level_SEA_parity:.3f}, Char-Level BPE: {g_total_char_level_SEA_parity:.3f}, BLT: {g_total_blt:.3f}"
    )

    # Global ratios across all languages
    total_bytes_all = sum(token_totals_bytes.values())
    total_tokens_all = sum(token_totals_myte.values())
    total_SEA_parity_all = sum(SEA_token_totals_parity.values())
    total_byte_level_SEA_parity_all = sum(byte_level_SEA_token_totals_parity.values())
    total_char_level_SEA_parity_all = sum(char_level_SEA_token_totals_parity.values())
    total_blt_patches_all = sum(blt_token_patches.values())

    langs_with_bytes = len(token_totals_bytes)
    langs_with_tokens = len(token_totals_myte)
    langs_with_SEA_parity = len(SEA_token_totals_parity)
    langs_with_byte_level_SEA_parity = len(byte_level_SEA_token_totals_parity)
    langs_with_char_level_SEA_parity = len(char_level_SEA_token_totals_parity)
    langs_with_blt = len(blt_token_patches)

    utf8_lines_per_byte_all = (langs_with_bytes * LINES) / total_bytes_all if total_bytes_all > 0 else 0.0
    myte_lines_per_token_all = (langs_with_tokens * LINES) / total_tokens_all if total_tokens_all > 0 else 0.0
    SEA_parity_lines_per_token_all = (langs_with_SEA_parity * LINES) / total_SEA_parity_all if total_SEA_parity_all > 0 else 0.0
    byte_level_SEA_parity_lines_per_token_all = (langs_with_byte_level_SEA_parity * LINES) / total_byte_level_SEA_parity_all if total_byte_level_SEA_parity_all > 0 else 0.0
    char_level_SEA_parity_lines_per_token_all = (langs_with_char_level_SEA_parity * LINES) / total_char_level_SEA_parity_all if total_char_level_SEA_parity_all > 0 else 0.0
    blt_lines_per_patch_all = (langs_with_blt * LINES) / total_blt_patches_all if total_blt_patches_all > 0 else 0.0

    # Global averages across languages
    avg_utf8_len_per_line_all = (total_bytes_all / (langs_with_bytes * LINES)) if (langs_with_bytes * LINES) > 0 else 0.0
    avg_myte_len_per_line_all = (total_tokens_all / (langs_with_tokens * LINES)) if (langs_with_tokens * LINES) > 0 else 0.0
    avg_SEA_parity_len_per_line_all = (total_SEA_parity_all / (langs_with_SEA_parity * LINES)) if (langs_with_SEA_parity * LINES) > 0 else 0.0
    avg_byte_level_SEA_parity_len_per_line_all = (total_byte_level_SEA_parity_all / (langs_with_byte_level_SEA_parity * LINES)) if (langs_with_byte_level_SEA_parity * LINES) > 0 else 0.0
    avg_char_level_SEA_parity_len_per_line_all = (total_char_level_SEA_parity_all / (langs_with_char_level_SEA_parity * LINES)) if (langs_with_char_level_SEA_parity * LINES) > 0 else 0.0
    avg_blt_len_per_line_all = (total_blt_patches_all / (langs_with_blt * LINES)) if (langs_with_blt * LINES) > 0 else 0.0

    avg_utf8_parity_all = (sum(v / eng_bytes for v in token_totals_bytes.values()) / langs_with_bytes) if (eng_bytes and langs_with_bytes) else 0.0
    avg_myte_parity_all = (sum(v / eng_tokens for v in token_totals_myte.values()) / langs_with_tokens) if (eng_tokens and langs_with_tokens) else 0.0
    avg_SEA_parity_parity_all = (sum(v / eng_SEA_parity for v in SEA_token_totals_parity.values()) / langs_with_SEA_parity) if (eng_SEA_parity and langs_with_SEA_parity) else 0.0
    avg_byte_level_SEA_parity_parity_all = (sum(v / eng_byte_level_SEA_parity for v in byte_level_SEA_token_totals_parity.values()) / langs_with_byte_level_SEA_parity) if (eng_byte_level_SEA_parity and langs_with_byte_level_SEA_parity) else 0.0
    avg_char_level_SEA_parity_parity_all = (sum(v / eng_char_level_SEA_parity for v in char_level_SEA_token_totals_parity.values()) / langs_with_char_level_SEA_parity) if (eng_char_level_SEA_parity and langs_with_char_level_SEA_parity) else 0.0
    avg_blt_parity_all = (sum(v / eng_blt_patches for v in blt_token_patches.values()) / langs_with_blt) if (eng_blt_patches and langs_with_blt) else 0.0

    output_lines.append("=== Compression Rates across all languages (higher is better) ===")
    output_lines.append(f"UTF-8: {utf8_lines_per_byte_all:.6f}")
    output_lines.append(f"MYTE: {myte_lines_per_token_all:.6f}")
    output_lines.append(f"Parity-Aware BPE: {SEA_parity_lines_per_token_all:.6f}")
    output_lines.append(f"Byte-Level BPE: {byte_level_SEA_parity_lines_per_token_all:.6f}")
    output_lines.append(f"Char-Level BPE: {char_level_SEA_parity_lines_per_token_all:.6f}")
    output_lines.append(f"BLT: {blt_lines_per_patch_all:.6f}")

    output_lines.append("=== Average Parity (lower is better) ===")
    output_lines.append(f"UTF-8 Avg Parity (vs eng): {avg_utf8_parity_all:.6f}")
    output_lines.append(f"MYTE Avg Parity (vs eng): {avg_myte_parity_all:.6f}")
    output_lines.append(f"Parity-Aware BPE Avg Parity (vs eng): {avg_SEA_parity_parity_all:.6f}")
    output_lines.append(f"Byte-Level BPE Avg Parity (vs eng): {avg_byte_level_SEA_parity_parity_all:.6f}")
    output_lines.append(f"Char-Level BPE Avg Parity (vs eng): {avg_char_level_SEA_parity_parity_all:.6f}")
    output_lines.append(f"BLT Avg Parity (vs eng): {avg_blt_parity_all:.6f}")

    output_lines.append("=== Worst-case Parity (lower is better) ===")
    output_lines.append(f"UTF-8 Worst Parity (vs eng): {max(token_totals_bytes[lang] / eng_bytes for lang in token_totals_bytes)}")
    output_lines.append(f"MYTE Worst Parity (vs eng): {max(token_totals_myte[lang] / eng_tokens for lang in token_totals_myte)}")
    output_lines.append(f"Parity-Aware BPE Worst Parity (vs eng): {max(SEA_token_totals_parity[lang] / eng_SEA_parity for lang in SEA_token_totals_parity)}")
    output_lines.append(f"Byte-Level BPE Worst Parity (vs eng): {max(byte_level_SEA_token_totals_parity[lang] / eng_byte_level_SEA_parity for lang in byte_level_SEA_token_totals_parity)}")
    output_lines.append(f"Char-Level BPE Worst Parity (vs eng): {max(char_level_SEA_token_totals_parity[lang] / eng_char_level_SEA_parity for lang in char_level_SEA_token_totals_parity)}")
    output_lines.append(f"BLT Worst Parity (vs eng): {max(blt_token_patches[lang] / eng_blt_patches for lang in blt_token_patches)}")

    output_lines.append("=== Average tokens per Line (lower is better) ===")
    output_lines.append(f"UTF-8 Avg Tokens/Line: {avg_utf8_len_per_line_all:.6f}")
    output_lines.append(f"MYTE Avg Tokens/Line: {avg_myte_len_per_line_all:.6f}")
    output_lines.append(f"Parity-Aware BPE Avg Tokens/Line: {avg_SEA_parity_len_per_line_all:.6f}")
    output_lines.append(f"Byte-Level BPE Avg Tokens/Line: {avg_byte_level_SEA_parity_len_per_line_all:.6f}")
    output_lines.append(f"Char-Level BPE Avg Tokens/Line: {avg_char_level_SEA_parity_len_per_line_all:.6f}")
    output_lines.append(f"BLT Avg Tokens/Line: {avg_blt_len_per_line_all:.6f}")

    # Write output to file instead of printing
    with open("/home/kieron/fyp/tok_eval_output.log", "w", encoding="utf-8") as f:
        for line in output_lines:
            f.write(line + "\n\n")

if __name__ == "__main__":
    main()