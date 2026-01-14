# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import logging
import os
from datetime import datetime

import torch
from lm_eval import simple_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM

from bytelatent.args import (
    EvalArgs,
)
from bytelatent.checkpoint import consolidate_checkpoints
from bytelatent.config_parser import parse_args_to_pydantic_model
from bytelatent.data.file_util import get_fs
from bytelatent.distributed import (
    DistributedArgs,
    get_global_rank,
    get_world_size,
    setup_torch_distributed,
)
from bytelatent.generate import (
    PackedCausalTransformerGenerator,
    load_consolidated_model_and_tokenizer,
)

EVAL_FOLDER_NAME = "{:010d}"

logger = logging.getLogger()

def all_dicts_same(dict_list):
    if not dict_list:  # Check if the list is empty
        return True

    # Compare each dictionary to the first one
    first_dict = dict_list[0]
    return all(d == first_dict for d in dict_list)

class MockAccelerator:
    def gather(self, tensor):
        local = [torch.zeros_like(tensor) for _ in range(get_world_size())]
        torch.distributed.all_gather(local, tensor)
        return torch.stack(local)

    def wait_for_everyone(self):
        torch.distributed.barrier()

# Light wrapper around generator for lm-eval harness
class EvalHarnessLM(LM):
    def __init__(self, generator):
        super().__init__()
        self.generator = generator
        self.accelerator = MockAccelerator()
        self._rank = get_global_rank()
        self._world_size = get_world_size()
        self.device = generator.device

    def generate_until(self, requests: list[Instance]) -> list[str]:
        prompts, gen_args = zip(*[req.args for req in requests])
        assert all_dicts_same(gen_args), "Doesn't support different gen args for now"
        gen_args = gen_args[0]
        temperature = gen_args.get("temperature", 0.0)
        top_p = gen_args.get("top_p", None)
        top_k = gen_args.get("top_k", None)
        until = gen_args.get("until", [])

        self.generator.temperature = temperature
        self.generator.top_p = top_p
        self.generator.top_k = top_k
        self.generator.until = until
        generations, _, _ = self.generator.generate(prompts)
        filtered_gen = []
        for g in generations:
            for e in until:
                g = g.replace(e, "")
            filtered_gen.append(g)
        return filtered_gen

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        prompts, _ = zip(*[req.args for req in requests])
        inputs = [req.args[0] + req.args[1] for req in requests]
        max_gen_len = self.generator.max_gen_len
        # We temporarily lower max gen len
        self.generator.max_gen_len = 1
        _, lls, greedy = self.generator.generate_nocache(inputs)
        results = []
        for p, ll, gr in zip(prompts, lls, greedy):
            p_len = len(
                self.generator.tokenizer.encode(p, add_bos=False, add_eos=False)
            )
            results.append((ll[p_len:].sum().item(), gr[p_len:].all().item()))

        self.generator.max_gen_len = max_gen_len
        return results

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[float]:
        prompts = [req.args[0] for req in requests]
        max_gen_len = self.generator.max_gen_len
        # We temporarily lower max gen len
        self.generator.max_gen_len = 1
        _, lls, _ = self.generator.generate(prompts)
        results = []
        for ll in lls:
            results.append((ll.sum().item(),))
        self.generator.max_gen_len = max_gen_len

        return results


def launch_eval(eval_args: EvalArgs):
    assert eval_args.dump_dir is not None
    assert eval_args.ckpt_dir is not None

    timestamp = datetime.now().strftime("%m%d-%H%M")
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
    model.eval()
    logger.info("Model loaded")

    patcher_args = train_cfg.data.patcher_args.model_copy(deep=True)
    patcher_args.realtime_patching = True
    print("Loading entropy model and patcher")
    patcher_args.entropy_model_checkpoint_dir = os.path.join(
        "/home/kieron/blt/hf-weights", "entropy_model"
    )
    patcher = patcher_args.build()

    task_results = None
    if eval_args.run_tasks:
        assert eval_args.generator is not None
        assert eval_args.harness is not None
        generator = PackedCausalTransformerGenerator(
            eval_args.generator, model, tokenizer, patcher=patcher
        )
        wrap = EvalHarnessLM(generator)
        # TODO: This needs to be checked/sped up
        task_results = simple_evaluate(wrap, **eval_args.harness.model_dump())

    results = {"tasks": task_results}
    # TODO: Serial and Parallel yield slightly different number of bytes, debug this later,
    # leaving this log statement here to help with that.
    # logging.info("Rank: %s Results: %s", world_rank, results)

    if get_global_rank() == 0:
        with fs.open(os.path.join(dump_dir, "results.json"), "w") as f:
            f.write(json.dumps(results))
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
            f.write(json.dumps(timestamp | results) + "\n")
            f.flush()


def main():
    eval_args = parse_args_to_pydantic_model(EvalArgs, cli_args="apps/main/configs/eval.yaml")
    launch_eval(eval_args)

if __name__ == "__main__":
    main()
