import sys
sys.path.insert(0, "/home/kieron/fyp")

import time
from pathlib import Path
from typing import Dict, List, Any

import torch

# Set to True to enable testing for each tokenizer
TOKENIZERS_TO_TEST = {
    "utf8": False,
    "myte": True,
    "parity_aware_bpe": False,
    "blt": False,
}

# Tokenizer file paths
MYTE_DECOMPOSE_MAP_PATH = "/home/kieron/fyp/myte/byte_maps/decompose_map.json"
# MYTE_MERGE_MAP_PATH = "/home/kieron/fyp/myte/byte_maps/merge_map.json"
MYTE_MERGE_MAP_PATH = "/home/kieron/fyp/myte/mappings_decomposed_filtered/morf_map_wiki_4096.json"
PARITY_AWARE_BPE_PATH = "/home/kieron/fyp/parity_aware_bpe/45k_parity-aware_SEA_1m/tokenizer.json"
BLT_ENTROPY_MODEL_DIR = "/home/kieron/fyp/blt/hf-weights/entropy_model"
BLT_CHECKPOINT_PATH = "/home/kieron/fyp/blt/hf-weights/blt_1b"

LINES = 1012
EVAL_DIR = Path("/home/kieron/fyp/data/flores-plus_dev_devtest/")

MYTE_96 = [
    "afr_Latn",
    "amh_Ethi",
    "arb_Arab",
    "azj_Latn",
    "bel_Cyrl",
    "bul_Cyrl",
    "ben_Beng",
    "cat_Latn",
    "ceb_Latn",
    "ces_Latn",
    "cym_Latn",
    "dan_Latn",
    "deu_Latn",
    "ell_Grek",
    "eng_Latn",
    "epo_Latn",
    "spa_Latn",
    "ekk_Latn",
    "eus_Latn",
    "pes_Arab",
    "fin_Latn",
    "fao_Latn",
    "fra_Latn",
    "gle_Latn",
    "gla_Latn",
    "glg_Latn",
    "guj_Gujr",
    "hau_Latn",
    "heb_Hebr",
    "hin_Deva",
    "hat_Latn",
    "hun_Latn",
    "hye_Armn",
    "ind_Latn",
    "ibo_Latn",
    "isl_Latn",
    "ita_Latn",
    "jpn_Jpan",
    "jav_Latn",
    "kat_Geor",
    "kaz_Cyrl",
    "khm_Khmr",
    "kan_Knda",
    "kor_Hang",
    "kmr_Latn",
    "kir_Cyrl",
    "ltz_Latn",
    "lao_Laoo",
    "lit_Latn",
    "lvs_Latn",
    "plt_Latn",
    "mri_Latn",
    "mkd_Cyrl",
    "mal_Mlym",
    "khk_Cyrl",
    "mar_Deva",
    "zsm_Latn",
    "mlt_Latn",
    "mya_Mymr",
    "npi_Deva",
    "nld_Latn",
    "nob_Latn",
    "nya_Latn",
    "pan_Guru",
    "pol_Latn",
    "pbt_Arab",
    "por_Latn",
    "ron_Latn",
    "rus_Cyrl",
    "snd_Arab",
    "sin_Sinh",
    "slk_Latn",
    "slv_Latn",
    "smo_Latn",
    "sna_Latn",
    "som_Latn",
    "als_Latn",
    "srp_Cyrl",
    "sot_Latn",
    "sun_Latn",
    "swe_Latn",
    "swh_Latn",
    "tam_Taml",
    "tel_Telu",
    "tgk_Cyrl",
    "tha_Thai",
    "tur_Latn",
    "ukr_Cyrl",
    "urd_Arab",
    "uzn_Latn",
    "vie_Latn",
    "xho_Latn",
    "ydd_Hebr",
    "yor_Latn",
    "cmn_Hans",
    "zul_Latn",
]

PA_BPE_60 = [
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

PA_BPE_30 = [
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

SEA_11 = [
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
    "cmn_Hans",
]

VALID_LANGS = MYTE_96

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


def count_tokens(tokenizer: Any, lines: List[str]) -> int:
    if not lines:
        return 0
    tokens = tokenizer(lines, padding=False, add_special_tokens=False)
    return sum(len(sublist) for sublist in tokens["attention_mask"])


def count_parity_tokens(tokenizer: Any, lines: List[str]) -> int:
    if not lines:
        return 0
    tokens = tokenizer.encode_batch(lines)
    return sum(len(enc.ids) for enc in tokens)


def count_blt_patches(tokenizer: Any, patcher: Any, lines: List[str]) -> int:
    """
    Counts patches used by BLT.
    NOTE: Patcher is passed in to avoid reloading the model on every call.
    """
    total_patches = 0

    for prompt in lines:
        token_ids = tokenizer.encode(prompt)
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()

        if not token_ids:
            continue

        tokens_tensor = torch.tensor([token_ids], dtype=torch.long, device="cuda:0")
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
    enabled = {k for k, v in TOKENIZERS_TO_TEST.items() if v}
    if not enabled:
        raise ValueError("No tokenizers enabled. Set at least one entry in TOKENIZERS_TO_TEST to True.")

    print(f"Enabled tokenizers: {', '.join(sorted(enabled))}")

    # Load only used tokenizers
    myte_tokenizer = None
    byte_tokenizer = None
    parity_aware_bpe_tokenizer = None
    blt_tokenizer = None
    blt_patcher = None

    if "utf8" in enabled:
        byte_tokenizer = ByteTokenizer()

    if "myte" in enabled:
        from myte.src.myt5.myt5_tokenizer import MyT5Tokenizer
        myte_tokenizer = MyT5Tokenizer(
            decompose_map=MYTE_DECOMPOSE_MAP_PATH,
            merge_map=MYTE_MERGE_MAP_PATH
        )

    if "parity_aware_bpe" in enabled:
        from tokenizers import Tokenizer
        parity_aware_bpe_tokenizer = Tokenizer.from_file(
            PARITY_AWARE_BPE_PATH
        )

    if "blt" in enabled:
        from bytelatent.generate import load_consolidated_model_and_tokenizer
        from bytelatent.data.patcher import PatcherArgs, PatchingModeEnum

        entropy_model_dir = Path(BLT_ENTROPY_MODEL_DIR)
        checkpoint_path = Path(BLT_CHECKPOINT_PATH)

        print("Loading BLT model...")
        _, blt_tokenizer, _ = load_consolidated_model_and_tokenizer(checkpoint_path)

        print("Initializing BLT Patcher...")
        patcher_args = PatcherArgs(
            patching_mode=PatchingModeEnum.entropy,
            realtime_patching=True,
            entropy_model_checkpoint_dir=str(entropy_model_dir),
            patching_device="cuda:0",
            device="cuda:0",
        )
        blt_patcher = patcher_args.build()

    # Metrics dicts
    tokens_per_lang_bytes: Dict[str, int] = {}
    avg_tokens_bytes: Dict[str, float] = {}
    avg_parity_bytes: Dict[str, float] = {}

    tokens_per_lang_myte: Dict[str, int] = {}
    avg_tokens_myte: Dict[str, float] = {}
    avg_parity_myte: Dict[str, float] = {}

    tokens_per_lang_pa_bpe: Dict[str, int] = {}
    avg_tokens_pa_bpe: Dict[str, float] = {}
    avg_parity_pa_bpe: Dict[str, float] = {}

    tokens_per_lang_blt: Dict[str, int] = {}
    avg_tokens_blt: Dict[str, float] = {}
    avg_parity_blt: Dict[str, float] = {}

    if not EVAL_DIR.is_dir():
        raise FileNotFoundError(f"Directory not found: {EVAL_DIR}")

    output_lines: List[str] = []
    output_lines.append(f"Evaluation started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    output_lines.append(f"Enabled tokenizers: {', '.join(sorted(enabled))}")

    files_to_process = [
        entry for entry in sorted(EVAL_DIR.iterdir())
        if entry.is_file() and entry.name.rsplit(".", 1)[0] in VALID_LANGS
    ]

    for idx, entry in enumerate(files_to_process, 1):
        code = entry.name.rsplit(".", 1)[0]
        print(f"[{idx}/{len(files_to_process)}] Processing language: {code}")

        lines = read_lines(entry, LINES)
        num_lines = len(lines)
        if num_lines == 0:
            continue

        output_lines.append(f"\n--- Language: {code} ---")
        per_lang_parts = []

        if "utf8" in enabled:
            total_tokens_bytes = count_tokens(byte_tokenizer, lines)
            tokens_per_lang_bytes[code] = total_tokens_bytes
            avg_tokens_bytes[code] = total_tokens_bytes / num_lines
            per_lang_parts.append(f"UTF-8 bytes={total_tokens_bytes}")

        if "myte" in enabled:
            total_tokens_myte = count_tokens(myte_tokenizer, lines)
            tokens_per_lang_myte[code] = total_tokens_myte
            avg_tokens_myte[code] = total_tokens_myte / num_lines
            per_lang_parts.append(f"MYTE tokens={total_tokens_myte}")

        if "parity_aware_bpe" in enabled:
            total_tokens_pa_bpe = count_parity_tokens(parity_aware_bpe_tokenizer, lines)
            tokens_per_lang_pa_bpe[code] = total_tokens_pa_bpe
            avg_tokens_pa_bpe[code] = total_tokens_pa_bpe / num_lines
            per_lang_parts.append(f"Parity-Aware BPE tokens={total_tokens_pa_bpe}")

        if "blt" in enabled:
            total_tokens_blt = count_blt_patches(blt_tokenizer, blt_patcher, lines)
            tokens_per_lang_blt[code] = total_tokens_blt
            avg_tokens_blt[code] = total_tokens_blt / num_lines
            per_lang_parts.append(f"BLT patches={total_tokens_blt}")

        output_lines.append(", ".join(per_lang_parts))

    # English baselines (only for enabled tokenizers)
    eng_bytes = tokens_per_lang_bytes.get("eng_Latn") if "utf8" in enabled else None
    eng_myte = tokens_per_lang_myte.get("eng_Latn") if "myte" in enabled else None
    eng_pa_bpe = tokens_per_lang_pa_bpe.get("eng_Latn") if "parity_aware_bpe" in enabled else None
    eng_blt = tokens_per_lang_blt.get("eng_Latn") if "blt" in enabled else None

    # Per-language parity + compression stats (only for enabled tokenizers)
    all_langs = sorted(set().union(
        avg_tokens_bytes.keys(),
        avg_tokens_myte.keys(),
        avg_tokens_pa_bpe.keys(),
        avg_tokens_blt.keys(),
    ))

    for lang in all_langs:
        output_lines.append(f"\n--- Language: {lang} ---")

        if "utf8" in enabled and lang in avg_tokens_bytes and eng_bytes:
            parity = tokens_per_lang_bytes[lang] / eng_bytes
            avg_parity_bytes[lang] = parity
            lines_per_token = 1 / avg_tokens_bytes[lang] if avg_tokens_bytes[lang] > 0 else 0.0
            parts = []
            parts.append(f"UTF-8 parity: {parity:.2f}")
            parts.append(f"UTF-8 average bytes per line: {avg_tokens_bytes[lang]:.2f}")
            parts.append(f"UTF-8 compression rate: {lines_per_token:.4f}")
            output_lines.append(", ".join(parts))

        if "myte" in enabled and lang in avg_tokens_myte and eng_myte:
            parity = tokens_per_lang_myte[lang] / eng_myte
            avg_parity_myte[lang] = parity
            lines_per_token = 1 / avg_tokens_myte[lang] if avg_tokens_myte[lang] > 0 else 0.0
            parts = []
            parts.append(f"MYTE parity: {parity:.2f}")
            parts.append(f"MYTE average tokens per line: {avg_tokens_myte[lang]:.2f}")
            parts.append(f"MYTE compression rate: {lines_per_token:.4f}")
            output_lines.append(", ".join(parts))
        if "parity_aware_bpe" in enabled and lang in avg_tokens_pa_bpe and eng_pa_bpe:
            parity = tokens_per_lang_pa_bpe[lang] / eng_pa_bpe
            avg_parity_pa_bpe[lang] = parity
            lines_per_token = 1 / avg_tokens_pa_bpe[lang] if avg_tokens_pa_bpe[lang] > 0 else 0.0
            parts = []
            parts.append(f"Parity-Aware BPE parity: {parity:.2f}")
            parts.append(f"Parity-Aware BPE average tokens per line: {avg_tokens_pa_bpe[lang]:.2f}")
            parts.append(f"Parity-Aware BPE compression rate: {lines_per_token:.4f}")
            output_lines.append(", ".join(parts))

        if "blt" in enabled and lang in avg_tokens_blt and eng_blt:
            parity = tokens_per_lang_blt[lang] / eng_blt
            avg_parity_blt[lang] = parity
            lines_per_token = 1 / avg_tokens_blt[lang] if avg_tokens_blt[lang] > 0 else 0.0
            parts = []
            parts.append(f"BLT parity: {parity:.2f}")
            parts.append(f"BLT average patches per line: {avg_tokens_blt[lang]:.2f}")
            parts.append(f"BLT compression rate: {lines_per_token:.4f}")
            output_lines.append(", ".join(parts))

    # Gini coefficients (only for enabled tokenizers)
    output_lines.append("\n--- Gini Coefficients (across languages) ---")
    gini_parts: List[str] = []
    if "utf8" in enabled:
        gini_parts.append(f"UTF-8: {gini(list(avg_tokens_bytes.values())):.3f}")
    if "myte" in enabled:
        gini_parts.append(f"MYTE: {gini(list(avg_tokens_myte.values())):.3f}")
    if "parity_aware_bpe" in enabled:
        gini_parts.append(f"Parity-Aware BPE: {gini(list(avg_tokens_pa_bpe.values())):.3f}")
    if "blt" in enabled:
        gini_parts.append(f"BLT: {gini(list(avg_tokens_blt.values())):.3f}")
    output_lines.append(", ".join(gini_parts) if gini_parts else "No enabled tokenizers produced data.")

    def _combined_avg(d: Dict[str, float]) -> float:
        return (sum(d.values()) / len(d)) if d else 0.0

    def _inv(x: float) -> float:
        return (1 / x) if x > 0 else 0.0

    # Global averages across languages (only for enabled)
    output_lines.append("\n--- Average tokens per line (lower is better) ---")
    if "utf8" in enabled:
        output_lines.append(f"UTF-8: {_combined_avg(avg_tokens_bytes):.2f}")
    if "myte" in enabled:
        output_lines.append(f"MYTE: {_combined_avg(avg_tokens_myte):.2f}")
    if "parity_aware_bpe" in enabled:
        output_lines.append(f"Parity-Aware BPE: {_combined_avg(avg_tokens_pa_bpe):.2f}")
    if "blt" in enabled:
        output_lines.append(f"BLT: {_combined_avg(avg_tokens_blt):.2f}")

    output_lines.append("\n--- Compression Rates across all languages (higher is better) ---")
    if "utf8" in enabled:
        output_lines.append(f"UTF-8: {_inv(_combined_avg(avg_tokens_bytes)):.4f}")
    if "myte" in enabled:
        output_lines.append(f"MYTE: {_inv(_combined_avg(avg_tokens_myte)):.4f}")
    if "parity_aware_bpe" in enabled:
        output_lines.append(f"Parity-Aware BPE: {_inv(_combined_avg(avg_tokens_pa_bpe)):.4f}")
    if "blt" in enabled:
        output_lines.append(f"BLT: {_inv(_combined_avg(avg_tokens_blt)):.4f}")

    output_lines.append("\n--- Average Parity vs English (lower is better) ---")
    if "utf8" in enabled:
        output_lines.append(f"UTF-8: {_combined_avg(avg_parity_bytes):.2f}" if eng_bytes else "UTF-8: English missing")
    if "myte" in enabled:
        output_lines.append(f"MYTE: {_combined_avg(avg_parity_myte):.2f}" if eng_myte else "MYTE: English missing")
    if "parity_aware_bpe" in enabled:
        output_lines.append(f"Parity-Aware BPE: {_combined_avg(avg_parity_pa_bpe):.2f}" if eng_pa_bpe else "Parity-Aware BPE: English missing")
    if "blt" in enabled:
        output_lines.append(f"BLT: {_combined_avg(avg_parity_blt):.2f}" if eng_blt else "BLT: English missing")

    output_lines.append("\n--- Worst-case Parity vs English (lower is better) ---")
    if "utf8" in enabled:
        output_lines.append(f"UTF-8: {max(avg_parity_bytes.values()):.2f}" if eng_bytes and avg_parity_bytes else "UTF-8: English missing")
    if "myte" in enabled:
        output_lines.append(f"MYTE: {max(avg_parity_myte.values()):.2f}" if eng_myte and avg_parity_myte else "MYTE: English missing")
    if "parity_aware_bpe" in enabled:
        output_lines.append(f"Parity-Aware BPE: {max(avg_parity_pa_bpe.values()):.2f}" if eng_pa_bpe and avg_parity_pa_bpe else "Parity-Aware BPE: English missing")
    if "blt" in enabled:
        output_lines.append(f"BLT: {max(avg_parity_blt.values()):.2f}" if eng_blt and avg_parity_blt else "BLT: English missing")

    # Write output to file instead of printing
    timestamp = time.strftime("%b%d-%H%M", time.localtime())
    output_path = Path(f"/home/kieron/fyp/output_tokenizer_eval_{timestamp}.log")
    with open(output_path, "w", encoding="utf-8") as f:
        for line in output_lines:
            f.write(line + "\n")


if __name__ == "__main__":
    main()