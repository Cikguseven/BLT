import sys
sys.path.insert(0, "/home/kieron/fyp")

from pathlib import Path
from typing import List, Dict, Any

# Configuration
TOKENIZERS_TO_TEST = {
    "utf8": True,
    "myte": True,
    "parity_aware_bpe": True,
    "blt": True,
}

# File paths (same as original)
MYTE_DECOMPOSE_MAP_PATH = "/home/kieron/fyp/myte/byte_maps/decompose_map.json"
MYTE_MERGE_MAP_PATH = "/home/kieron/fyp/myte/mappings_decomposed_filtered/morf_map_mc4_8192_equal.json"
PARITY_AWARE_BPE_PATH = "/home/kieron/fyp/parity_aware_bpe/90k_byte-level_SEA_1m/tokenizer.json"
BLT_ENTROPY_MODEL_DIR = "/home/kieron/fyp/blt/blt-entropy/checkpoints/SEA_1M_proportional/consolidated"
BLT_CHECKPOINT_PATH = "/home/kieron/fyp/blt/hf-weights/blt_1b"

# Test sentences in different languages
TEST_SENTENCES = {
    "eng_Latn": "Hello, how are you today?",
    "cmn_Hans": "你好，今天怎么样？",
    "tha_Thai": "สวัสดี วันนี้เป็นอย่างไรบ้าง",
    "vie_Latn": "Xin chào, hôm nay bạn thế nào?",
}


def print_separator(char="-", length=80):
    print(char * length)


def load_tokenizers():
    """Load all enabled tokenizers."""
    tokenizers = {}

    if TOKENIZERS_TO_TEST.get("utf8"):
        print("Loading UTF-8 byte tokenizer...")
        # UTF-8 is always available (no external loading)
        tokenizers["utf8"] = "loaded"

    if TOKENIZERS_TO_TEST.get("myte"):
        print("Loading MYTE tokenizer...")
        try:
            from myte.src.myt5.myt5_tokenizer import MyT5Tokenizer
            tokenizers["myte"] = MyT5Tokenizer(
                decompose_map=MYTE_DECOMPOSE_MAP_PATH,
                merge_map=MYTE_MERGE_MAP_PATH
            )
            print("✓ MYTE loaded successfully")
        except Exception as e:
            print(f"✗ MYTE failed: {e}")

    if TOKENIZERS_TO_TEST.get("parity_aware_bpe"):
        print("Loading Parity-Aware BPE tokenizer...")
        try:
            from tokenizers import Tokenizer
            tokenizers["parity_aware_bpe"] = Tokenizer.from_file(PARITY_AWARE_BPE_PATH)
            print("✓ Parity-Aware BPE loaded successfully")
        except Exception as e:
            print(f"✗ Parity-Aware BPE failed: {e}")

    if TOKENIZERS_TO_TEST.get("blt"):
        print("Loading BLT tokenizer...")
        try:
            import torch
            from bytelatent.generate import load_consolidated_model_and_tokenizer
            from bytelatent.data.patcher import PatcherArgs, PatchingModeEnum

            _, tokenizer, _ = load_consolidated_model_and_tokenizer(Path(BLT_CHECKPOINT_PATH))

            patcher_args = PatcherArgs(
                patching_mode=PatchingModeEnum.entropy,
                realtime_patching=True,
                entropy_model_checkpoint_dir=str(BLT_ENTROPY_MODEL_DIR),
                patching_device="cuda:0",
                device="cuda:0",
            )
            patcher = patcher_args.build()

            tokenizers["blt"] = {"tokenizer": tokenizer, "patcher": patcher}
            print("✓ BLT loaded successfully")
        except Exception as e:
            print(f"✗ BLT failed: {e}")

    return tokenizers


def tokenize_utf8(text: str) -> List[str]:
    """Tokenize text into UTF-8 bytes."""
    byte_sequence = text.encode("utf-8")
    return [f"0x{b:02x}" for b in byte_sequence]


def tokenize_myte(tokenizer: Any, text: str) -> List[str]:
    """Tokenize with MYTE and return token strings."""
    out = tokenizer([text], padding=False, add_special_tokens=False)
    token_ids = out["input_ids"][0]
    # Convert back to tokens (strings)
    tokens = [tokenizer.convert_ids_to_tokens(tid) for tid in token_ids]
    return tokens


def tokenize_parity_bpe(tokenizer: Any, text: str) -> List[str]:
    """Tokenize with Parity-Aware BPE and return token strings."""
    encoding = tokenizer.encode(text)
    return encoding.tokens


def tokenize_blt(tokenizer_dict: Dict[str, Any], text: str) -> List[str]:
    """Tokenize with BLT and return patch information."""
    import torch

    tokenizer = tokenizer_dict["tokenizer"]
    patcher = tokenizer_dict["patcher"]

    token_ids = tokenizer.encode(text)
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()

    if not token_ids:
        return []

    tokens_tensor = torch.tensor([token_ids], dtype=torch.long, device="cuda:0")
    patch_lengths, _ = patcher.patch(tokens_tensor, include_next_token=False)

    # Return patch information
    patches = patch_lengths.squeeze(0).tolist()
    return [f"patch[{i}]:{length}" for i, length in enumerate(patches)]


def run_sanity_checks(tokenizers: Dict[str, Any]):
    """Run sanity checks on all tokenizers."""

    for lang_code, sentence in TEST_SENTENCES.items():
        print_separator("=")
        print(f"Language: {lang_code}")
        print(f"Text: {sentence}")
        print(f"Text length: {len(sentence)} characters, {len(sentence.encode('utf-8'))} bytes")
        print_separator()

        # UTF-8
        if "utf8" in tokenizers:
            tokens = tokenize_utf8(sentence)
            print(f"\\n[UTF-8]")
            print(f"Token count: {len(tokens)}")
            print(f"First 20 tokens: {tokens[:20]}")
            if len(tokens) > 20:
                print(f"... ({len(tokens) - 20} more)")

        # MYTE
        if "myte" in tokenizers:
            try:
                tokens = tokenize_myte(tokenizers["myte"], sentence)
                print(f"\\n[MYTE]")
                print(f"Token count: {len(tokens)}")
                print(f"First 20 tokens: {tokens[:20]}")
                if len(tokens) > 20:
                    print(f"... ({len(tokens) - 20} more)")
            except Exception as e:
                print(f"\\n[MYTE] Error: {e}")

        # Parity-Aware BPE
        if "parity_aware_bpe" in tokenizers:
            try:
                tokens = tokenize_parity_bpe(tokenizers["parity_aware_bpe"], sentence)
                print(f"\\n[Parity-Aware BPE]")
                print(f"Token count: {len(tokens)}")
                print(f"First 20 tokens: {tokens[:20]}")
                if len(tokens) > 20:
                    print(f"... ({len(tokens) - 20} more)")
            except Exception as e:
                print(f"\\n[Parity-Aware BPE] Error: {e}")

        # BLT
        if "blt" in tokenizers:
            try:
                patches = tokenize_blt(tokenizers["blt"], sentence)
                print(f"\\n[BLT]")
                print(f"Patch count: {len(patches)}")
                print(f"First 20 patches: {patches[:20]}")
                if len(patches) > 20:
                    print(f"... ({len(patches) - 20} more)")
            except Exception as e:
                print(f"\\n[BLT] Error: {e}")

        print()


def main():
    print("=" * 80)
    print("TOKENIZER SANITY CHECK")
    print("=" * 80)
    print()

    # Load tokenizers
    tokenizers = load_tokenizers()
    print(f"\\nLoaded {len(tokenizers)} tokenizer(s)")
    print()

    # Run sanity checks
    run_sanity_checks(tokenizers)

    print_separator("=")
    print("Sanity check complete!")


if __name__ == "__main__":
    main()
