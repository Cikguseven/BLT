from pathlib import Path

import torch

from bytelatent.generate import load_consolidated_model_and_tokenizer
from bytelatent.data.patcher import PatcherArgs, PatchingModeEnum


def main(prompts: list[str]):
    entropy_model_dir = Path("/home/kieron/fyp/blt/hf-weights/entropy_model")
    checkpoint_path = Path("/home/kieron/fyp/blt/hf-weights/blt_1b")

    _, tokenizer, _ = load_consolidated_model_and_tokenizer(checkpoint_path)

    patcher_args = PatcherArgs(
        patching_mode=PatchingModeEnum.entropy,
        realtime_patching=True,
        entropy_model_checkpoint_dir=str(entropy_model_dir),
        patching_device="cuda",
        device="cuda",
    )
    patcher = patcher_args.build()

    for prompt in prompts:
        token_ids = tokenizer.encode(prompt)
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        print(f'Prompt: "{prompt}"')
        print(f"Token count: {len(token_ids)}")
        print(f"Token IDs: {token_ids}")

        tokens_tensor = torch.tensor([token_ids], dtype=torch.long, device="cuda")
        patch_lengths, entropies = patcher.patch(tokens_tensor, include_next_token=False)
        patch_lengths_list = patch_lengths.squeeze(0).tolist()
        print(f"Patch lengths (entropy-based): {patch_lengths_list}")


        if entropies is not None:
            entropy_values = entropies.squeeze(0).tolist()
            print(f"Per-token entropies: {entropy_values}")
        print("-" * 40)


if __name__ == "__main__":
    prompts = [
        "We now have 4-month-old mice that are non-diabetic that used to be diabetic, he added.",
        "The quick brown fox jumps over the lazy dog.",
        "ByteLatent is a novel tokenizer for language models."
    ]
    main(prompts)
