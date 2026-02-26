import sys
sys.path.insert(0, "/scratch/Projects/CFP-01/CFP01-CF-060/kieron")

import time
from pathlib import Path
from typing import List, Any, Set, Tuple

import torch

TOKENIZERS_TO_TEST = {
    "myte": False,
    "parity_aware_bpe": True,
    "blt": False,
}

# Tokenizer file paths
MYTE_DECOMPOSE_MAP_PATH = "/home/kieron/fyp/myte/byte_maps/decompose_map.json"
MYTE_MERGE_MAP_PATH = "/home/kieron/fyp/myte/mappings_decomposed_filtered/morf_map_mc4_8192.json"
PARITY_AWARE_BPE_PATH = "/home/kieron/fyp/parity_aware_bpe/90k_parity-aware_SEA_1m/tokenizer.json"
BLT_ENTROPY_MODEL_DIR = "/home/kieron/fyp/blt/blt-entropy-mc4-1M-original"
BLT_CHECKPOINT_PATH = "/home/kieron/fyp/blt/hf-weights/blt_1b"

LINES = 1012
EVAL_DIR = Path("/scratch/Projects/CFP-01/CFP01-CF-060/kieron/data/flores-plus_dev_devtest")

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

VALID_LANGS = SEA_11


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


def count_bytes(lines: List[str]) -> int:
    return sum(len(line.encode("utf-8")) for line in lines)


def get_unique_myte_tokens(tokenizer: Any, lines: List[str]) -> Set[int]:
    if not lines:
        return unique_bytes, unique_morphemes

    out = tokenizer(lines, padding=False, add_special_tokens=False)

    for ids in out["input_ids"]:
        # Track every unique byte used
        unique_bytes.update(ids)

        # Parse the sequence to extract unique multibyte morphemes
        i = 0
        while i < len(ids):
            b = ids[i]
            # Check if this byte is a MYTE morpheme leading byte (0x42 to 0x5A)
            if 66 <= b <= 90:
                morpheme_chunk = [b]
                i += 1
                # Collect any standard UTF-8 continuation bytes (0x80 to 0xBF)
                while i < len(ids) and 128 <= ids[i] <= 191:
                    morpheme_chunk.append(ids[i])
                    i += 1
                unique_morphemes.add(tuple(morpheme_chunk))
            else:
                i += 1

    return unique_bytes, unique_morphemes


def get_unique_parity_tokens(tokenizer: Any, lines: List[str]) -> Set[int]:
    if not lines:
        return set()
    encs = tokenizer.encode_batch(lines)
    unique = set()
    for enc in encs:
        unique.update(enc.ids)
    return unique


def get_unique_blt_patches(tokenizer: Any, patcher: Any, lines: List[str]) -> Set[Tuple[int, ...]]:
    unique_patches = set()
    for prompt in lines:
        token_ids = tokenizer.encode(prompt)
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if not token_ids:
            continue

        tokens_tensor = torch.tensor([token_ids], dtype=torch.long, device="cuda:0")
        patch_lengths, _ = patcher.patch(tokens_tensor, include_next_token=False)

        lengths = patch_lengths.squeeze(0).tolist()

        start_idx = 0
        for length in lengths:
            patch = tuple(token_ids[start_idx : start_idx + length])
            unique_patches.add(patch)
            start_idx += length

    return unique_patches


def main() -> None:
    enabled = {k for k, v in TOKENIZERS_TO_TEST.items() if v}
    if not enabled:
        raise ValueError("No tokenizers enabled. Set at least one entry in TOKENIZERS_TO_TEST to True.")

    print(f"Enabled tokenizers: {', '.join(sorted(enabled))}")

    myte_tokenizer = None
    parity_aware_bpe_tokenizer = None
    blt_tokenizer = None
    blt_patcher = None

    if "myte" in enabled:
        from myte.src.myt5.myt5_tokenizer import MyT5Tokenizer
        myte_tokenizer = MyT5Tokenizer(
            decompose_map=MYTE_DECOMPOSE_MAP_PATH,
            merge_map=MYTE_MERGE_MAP_PATH
        )

    if "parity_aware_bpe" in enabled:
        from tokenizers import Tokenizer
        parity_aware_bpe_tokenizer = Tokenizer.from_file(PARITY_AWARE_BPE_PATH)

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

    if not EVAL_DIR.is_dir():
        raise FileNotFoundError(f"Directory not found: {EVAL_DIR}")

    output_lines: List[str] = []
    output_lines.append(f"Evaluation started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    output_lines.append(f"Enabled tokenizers: {', '.join(sorted(enabled))}")

    global_unique_myte: Set[int] = set()
    global_unique_pa_bpe: Set[int] = set()
    global_unique_blt_patches: Set[Tuple[int, ...]] = set()
    global_total_bytes: int = 0  # <-- NEW

    files_to_process = [
        entry for entry in sorted(EVAL_DIR.iterdir())
        if entry.is_file() and entry.name.rsplit(".", 1)[0] in VALID_LANGS and entry.suffix == ".devtest"
    ]

    for idx, entry in enumerate(files_to_process, 1):
        code = entry.name.rsplit(".", 1)[0]
        print(f"[{idx}/{len(files_to_process)}] Processing language: {code}")

        lines = read_lines(entry, LINES)
        if not lines:
            continue

        output_lines.append(f"\n--- Language: {code} ---")

        # Byte count for this language
        lang_bytes = count_bytes(lines)           # <-- NEW
        global_total_bytes += lang_bytes           # <-- NEW
        output_lines.append(f"Total bytes: {lang_bytes}")  # <-- NEW

        if "myte" in enabled:
            unique_bytes, unique_morphemes = get_unique_myte_tokens(myte_tokenizer, lines)
            global_unique_myte_bytes.update(unique_bytes)
            global_unique_myte_morphemes.update(unique_morphemes)
            output_lines.append(f"MYTE distinct bytes: {len(unique_bytes)}")
            output_lines.append(f"MYTE distinct morphemes: {len(unique_morphemes)}")

        if "parity_aware_bpe" in enabled:
            unique_tokens = get_unique_parity_tokens(parity_aware_bpe_tokenizer, lines)
            global_unique_pa_bpe.update(unique_tokens)
            output_lines.append(f"Parity-Aware BPE distinct tokens: {len(unique_tokens)}")

        if "blt" in enabled:
            unique_patches = get_unique_blt_patches(blt_tokenizer, blt_patcher, lines)
            global_unique_blt_patches.update(unique_patches)
            output_lines.append(f"BLT distinct patches: {len(unique_patches)}")

    output_lines.append("\n========================================================")
    output_lines.append("GLOBAL DISTINCT TOKENS/PATCHES (Across all languages & sentences)")
    output_lines.append("========================================================")

    output_lines.append(f"Total bytes (all languages): {global_total_bytes}")  # <-- NEW

    if "myte" in enabled:
        output_lines.append(f"MYTE total distinct bytes: {len(global_unique_myte_bytes)}")
        output_lines.append(f"MYTE total distinct morphemes: {len(global_unique_myte_morphemes)}")
    if "parity_aware_bpe" in enabled:
        output_lines.append(f"Parity-Aware BPE total distinct tokens: {len(global_unique_pa_bpe)}")
    if "blt" in enabled:
        output_lines.append(f"BLT total distinct patches: {len(global_unique_blt_patches)}")

    timestamp = time.strftime("%b%d-%H%M", time.localtime())
    output_path = Path(f"/scratch/Projects/CFP-01/CFP01-CF-060/kieron/output_distinct_tokens_{timestamp}.log")
    with open(output_path, "w", encoding="utf-8") as f:
        for line in output_lines:
            f.write(line + "\n")

    print(f"\nFinished. Results written to {output_path}")


if __name__ == "__main__":
    main()
