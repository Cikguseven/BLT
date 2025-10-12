import os

import typer
from huggingface_hub import snapshot_download


def main():
    if not os.path.exists("hf-blt-1b"):
        os.makedirs("hf-blt-1b")

    snapshot_download(
        "facebook/blt-1b", local_dir="hf-blt-1b", ignore_patterns="entropy_model/"
    )


if __name__ == "__main__":
    typer.run(main)
