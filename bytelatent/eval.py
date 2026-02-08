# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import logging
import math
import os
import sys
from datetime import datetime

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
from lm_eval import simple_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from torch.nn import functional as F

from bytelatent.args import EvalArgs
from bytelatent.checkpoint import consolidate_checkpoints
from bytelatent.config_parser import parse_args_to_pydantic_model
from bytelatent.data.file_util import get_fs
from bytelatent.data.iterators.arrow_iterator import ArrowFileIterator
from bytelatent.data.iterators.limit_iterator import LimitIterator
from bytelatent.data.iterators.packing_iterator import (
    PackingArgs,
    PackingIterator
)
from bytelatent.data.iterators.preprocess_iterator import PreprocessIterator
from bytelatent.data.iterators.sequence_iterator import (
    SequenceIterator,
    SequencePackingArgs,
)
from bytelatent.data.patcher import PatcherArgs
from bytelatent.distributed import (
    DistributedArgs,
    dist_sum,
    get_global_rank,
    get_world_size,
    setup_torch_distributed,
    to_py_num,
)
from bytelatent.generate import (
    load_consolidated_model_and_tokenizer,
)
from bytelatent.generate_blt import generate_nocache
from bytelatent.model.blt import ByteLatentTransformer
from bytelatent.blt_tokenizers.build_tokenizer import TokenizerArgs
from bytelatent.transformer import LMTransformer

logger = logging.getLogger()

def all_dicts_same(dict_list):
    if not dict_list:  # Check if the list is empty
        return True

    # Compare each dictionary to the first one
    first_dict = dict_list[0]
    return all(d == first_dict for d in dict_list)

class MockAccelerator:
    def gather(self, tensor):
        l = [torch.zeros_like(tensor) for _ in range(get_world_size())]
        torch.distributed.all_gather(l, tensor)
        return torch.stack(l)

    def wait_for_everyone(self):
        torch.distributed.barrier()

# Light wrapper around generator for lm-eval harness
class EvalHarnessLM(LM):
    def __init__(self, model, tokenizer, patcher):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.patcher = patcher
        self.accelerator = MockAccelerator()
        self._rank = get_global_rank()
        self._world_size = get_world_size()
        self.device = next(model.parameters()).device

        # Infer max sequence length
        if hasattr(model, "get_output_seq_len"):
            self.max_seq_len = model.get_output_seq_len()
        else:
            self.max_seq_len = 8192

    def generate_until(self, requests: list[Instance]) -> list[str]:
        results = [None] * len(requests)
        # Group by generation arguments
        groups = {}
        for i, req in enumerate(requests):
            prompt, gen_args = req.args
            # Convert dict to hashable tuple, handling list values like 'until'
            arg_key = tuple(sorted((k, tuple(v) if isinstance(v, list) else v)
                            for k, v in gen_args.items()))
            if arg_key not in groups:
                groups[arg_key] = []
            groups[arg_key].append((i, prompt, gen_args))

        for group in groups.values():
            indices, prompts, gen_args_list = zip(*group)
            gen_args = gen_args_list[0]

            temperature = gen_args.get("temperature", 0.0)
            use_sampling = temperature > 0.0
            temp_val = temperature if use_sampling else 1.0

            top_p = gen_args.get("top_p", 0.0)
            top_k = gen_args.get("top_k", 0)
            until = gen_args.get("until", [])

            max_gen_len = gen_args.get("max_gen_toks", 256)

            generated_tokens_list = generate_nocache(
                prompts=list(prompts),
                model=self.model,
                tokenizer=self.tokenizer,
                patcher=self.patcher,
                max_gen_len=max_gen_len,
                use_sampling=use_sampling,
                temp=temp_val,
                top_k=top_k,
                top_p=top_p,
                remove_prompts=True
            )

            for idx, ids in zip(indices, generated_tokens_list):
                text = self.tokenizer.decode(ids)
                for e in until:
                    if e in text:
                        text = text.split(e)[0]
                results[idx] = text

        return results

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        results = []
        for req in requests:
            prompt, continuation = req.args
            ctx_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
            cont_ids = self.tokenizer.encode(continuation, add_bos=False, add_eos=False)

            # If there's no continuation, define loglikelihood=0 and greedy=True
            if len(cont_ids) == 0:
                results.append((0.0, True))
                continue

            # Ensure the full sequence fits the model's max length.
            # We must keep at least 1 context token so the first continuation token is predictable.
            if len(cont_ids) >= self.max_seq_len:
                cont_ids = cont_ids[: max(1, self.max_seq_len - 1)]

            ctx_budget = self.max_seq_len - len(cont_ids)
            ctx_budget = max(1, ctx_budget)  # keep >= 1 context token always

            if len(ctx_ids) > ctx_budget:
                ctx_ids = ctx_ids[-ctx_budget:]

            # (Defensive) if encode ever returned empty, skip rather than crash kernels
            if len(ctx_ids) == 0:
                results.append((0.0, False))
                continue

            full_ids = ctx_ids + cont_ids
            tokens = torch.tensor([full_ids], device=self.device, dtype=torch.long)

            # Patching
            patch_lengths, _ = self.patcher.patch(tokens, include_next_token=False)
            if patch_lengths is not None:
                patch_lengths = patch_lengths.to(self.device)

            with torch.no_grad():
                # logits: [1, seq_len, vocab_size]
                logits = self.model(tokens, patch_lengths=patch_lengths)

            # Continuation is predicted starting from the last context token position.
            start_logit_idx = len(ctx_ids) - 1
            cont_len = len(cont_ids)
            end_logit_idx = start_logit_idx + cont_len

            # Clamp to what the model actually returned (extra safety)
            end_logit_idx = min(end_logit_idx, logits.size(1))
            cont_len = min(cont_len, end_logit_idx - start_logit_idx)

            if cont_len <= 0:
                results.append((0.0, False))
                continue

            relevant_logits = logits[0, start_logit_idx:end_logit_idx, :]
            target_tokens = tokens[0, len(ctx_ids) : len(ctx_ids) + cont_len]

            # Extra guard against any mismatch (prevents CUDA asserts)
            if relevant_logits.size(0) != target_tokens.numel():
                n = min(relevant_logits.size(0), target_tokens.numel())
                relevant_logits = relevant_logits[:n, :]
                target_tokens = target_tokens[:n]

            loss = F.cross_entropy(relevant_logits, target_tokens, reduction="sum")

            greedy_tokens = relevant_logits.argmax(dim=-1)
            is_greedy = (greedy_tokens == target_tokens).all().item()

            results.append((-loss.item(), is_greedy))

        return results

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[float]:
        results = []
        for req in requests:
            prompt = req.args[0]
            tokens_list = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
            tokens = torch.tensor([tokens_list], device=self.device, dtype=torch.long)

            patch_lengths, _ = self.patcher.patch(tokens, include_next_token=False)
            if patch_lengths is not None:
                patch_lengths = patch_lengths.to(self.device)

            with torch.no_grad():
                logits = self.model(tokens, patch_lengths=patch_lengths)

            shift_logits = logits[0, :-1, :].contiguous()
            shift_labels = tokens[0, 1:].contiguous()

            loss = F.cross_entropy(shift_logits, shift_labels, reduction="sum")
            results.append(-loss.item())

        return results


def launch_eval(eval_args: EvalArgs):
    assert eval_args.dump_dir is not None
    assert eval_args.ckpt_dir is not None

    timestamp = datetime.now().strftime("%b%d-%H%M")
    dump_dir = f"{eval_args.dump_dir}_{timestamp}"

    distributed_args = DistributedArgs()
    distributed_args.configure_world()
    if not torch.distributed.is_initialized():
        setup_torch_distributed(distributed_args)

    assert distributed_args.dp_shard == 1

    fs = get_fs(eval_args.ckpt_dir, s3_profile=eval_args.s3_profile)
    if (
        fs.exists(eval_args.ckpt_dir)
        and fs.exists(os.path.join(eval_args.ckpt_dir, "params.json"))
        and len(fs.glob(os.path.join(eval_args.ckpt_dir, "*.pth"))) != 0
    ):
        consolidate_path = eval_args.ckpt_dir
    else:
        if eval_args.consolidate_if_needed:
            logger.info(
                "Found a model checkpoint, but it has not been consolidated.... so consolidating the checkpoint"
            )
            consolidate_path = os.path.join(
                eval_args.ckpt_dir, eval_args.consolidate_folder
            )
            if not fs.exists(consolidate_path) and get_global_rank() == 0:
                consolidate_path = consolidate_checkpoints(fs, eval_args.ckpt_dir)
            logger.info("Model consolidated to: %s", consolidate_path)
        else:
            raise ValueError(
                "Did not find a consolidated checkpoint and consolidate_if_needed is False"
            )

    fs.mkdirs(dump_dir, exist_ok=True)
    with fs.open(os.path.join(dump_dir, "config.yaml"), "w") as f:
        f.write(eval_args.model_dump_json())

    torch.distributed.barrier()
    logger.info("Loading model")
    model, tokenizer, train_cfg = load_consolidated_model_and_tokenizer(
        consolidate_path,
    )
    # pad_id = 0 if train_cfg.data.tokenizer_args.name == "bytes" else tokenizer.boe_id
    model.eval()
    logger.info("Model loaded")

    # Build Patcher
    patcher_args = train_cfg.data.patcher_args.model_copy(deep=True)
    patcher_args.realtime_patching = True
    patcher_args.entropy_model_checkpoint_dir = eval_args.entropy_ckpt_dir
    patcher = patcher_args.build()

    task_results = None
    if eval_args.run_tasks:
        assert eval_args.harness is not None
        # Instantiate modified EvalHarnessLM
        wrap = EvalHarnessLM(model, tokenizer, patcher)
        # Add confirm_run_unsafe_code=True here
        task_results = simple_evaluate(
            wrap,
            **eval_args.harness.model_dump(),
            confirm_run_unsafe_code=True
        )
    results = {"tasks": task_results}
    # TODO: Serial and Parallel yield slightly different number of bytes, debug this later,
    # leaving this log statement here to help with that.
    # logging.info("Rank: %s Results: %s", world_rank, results)

    if get_global_rank() == 0:
        with fs.open(os.path.join(dump_dir, "results.json"), "w") as f:
            f.write(json.dumps(results, default=str))
        logger.info(f"All evaluation results: {results}")

    if eval_args.metric_log_dir and get_global_rank() == 0:
        metric_log_path = os.path.join(eval_args.metric_log_dir, "metrics.eval.jsonl")

        logger.info(f"Writing metric logs to {metric_log_path}")
        timestamp: dict[str, int | str] = {
            "created_at": datetime.utcnow().isoformat(),
        }
        if eval_args.global_step is not None:
            timestamp["global_step"] = eval_args.global_step
        with fs.open(metric_log_path, mode="a") as f:
            f.write(json.dumps(timestamp | results, default=str) + "\n")
            f.flush()

def main():
    eval_args = parse_args_to_pydantic_model(EvalArgs, cli_args="/home/kieron/fyp/blt/apps/main/configs/eval.yaml")
    launch_eval(eval_args)

if __name__ == "__main__":
    main()
