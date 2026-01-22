from typing import List
from pathlib import Path
from tokenizers import Tokenizer
import concurrent.futures
import os
import tiktoken

def read_lines(fp):
    lines: List[str] = []
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
    return lines

def count_parity_tokens(tokenizer, lines):
    if not lines:
        return 0
    tokens = tokenizer.encode_batch(lines)
    return sum(len(enc.ids) for enc in tokens)

def count_tiktokens(lines):
    if not lines:
        return 0
    enc = tiktoken.get_encoding("cl100k_base")
    total = 0
    for line in lines:
        ids = enc.encode(line)
        total += len(ids)
    return total

def process_file(path_str, model_path):
    # tokenizer = Tokenizer.from_file(model_path)
    p = Path(path_str)
    lines = read_lines(p)
    # c = count_parity_tokens(tokenizer, lines)
    c = count_tiktokens(lines)
    return str(p), c

def main(
    model_path,
    folder,
):
    root = Path(folder)
    files = sorted(list(root.glob("*.txt")))

    total = 0
    per_file = []

    max_workers = min(len(files) or 1, os.cpu_count() or 1)
    print(f"Processing {len(files)} files with {max_workers} processes...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_file, str(p), model_path) for p in files]

        for future in concurrent.futures.as_completed(futures):
            p, c = future.result()
            print(f"Processed {p}: {c}")
            per_file.append((p, c))
            total += c

    per_file.sort()
    for p, c in per_file:
        print(f"{c}\t{p}")
    print(f"\nTOTAL\t{total}")

if __name__ == "__main__":
    main(
        model_path="/home/kieron/fyp/parity_aware_bpe/45k_parity-aware_SEA_1m/tokenizer.json",
        folder="/home/kieron/fyp/data_mc4_60lang_100000_sentences"
    )
