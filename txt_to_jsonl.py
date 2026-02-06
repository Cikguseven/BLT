import json
import os
from pathlib import Path
from tqdm import tqdm

def count_lines_in_files(file_path):
    """Helper to get total line count"""
    total = 0
    print("Calculating total lines...")
    with open(file_path, "rb") as f:
        total += sum(1 for _ in f)
    return total

def txt_to_jsonl_split(input_file, output_dir, chunk_size=100_000):
    input_file = Path(input_file)
    output_dir = Path(output_dir)

    if not input_file.exists():
        print(f"Input file {input_file} does not exist.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Configuration for output filenames
    stem = "combined"
    suffix = ".jsonl"

    total_lines = count_lines_in_files(input_file)

    current_chunk_idx = 0
    lines_in_current_chunk = 0
    f_out = None

    try:
        with tqdm(total=total_lines, unit='lines', desc="Converting") as pbar:
           with input_file.open("r", encoding="utf-8", errors="ignore") as f_in:
                for line in f_in:
                    pbar.update(1)

                    line = line.strip()
                    if not line:
                        continue

                    # Open a new chunk file if we don't have one open
                    if f_out is None:
                        chunk_filename = f"{stem}.chunk.{current_chunk_idx:02d}{suffix}"
                        chunk_path = output_dir / chunk_filename
                        f_out = chunk_path.open("w", encoding="utf-8")

                    # Write the JSON object
                    obj = {"text": line}
                    f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    lines_in_current_chunk += 1

                    # Rotate file if chunk size limit is reached
                    if lines_in_current_chunk >= chunk_size:
                        f_out.close()
                        f_out = None
                        lines_in_current_chunk = 0
                        current_chunk_idx += 1

    finally:
        if f_out:
            f_out.close()

if __name__ == "__main__":
    txt_to_jsonl_split(
        input_file="/home/kieron/fyp/data/mc4_SEA_1000000_sentences_temp_0.3/combined.txt",
        output_dir="/home/kieron/fyp/data/mc4_SEA_1000000_sentences_temp_0.3/combined",
        chunk_size=100_000
    )
