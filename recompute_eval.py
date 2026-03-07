import json
import subprocess
import tempfile
import os

RESULTS_PATH = "/scratch/Projects/CFP-01/CFP01-CF-060/kieron/blt/dump-dir_Feb20-1301/results.json"
TIMEOUT = 10.0

def run_py(program: str) -> bool:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(program)
        fname = f.name
    try:
        r = subprocess.run(["python3", fname], capture_output=True, timeout=TIMEOUT)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        try:
            os.unlink(fname)
        except OSError:
            pass


def eval_humaneval(samples: list) -> tuple[float, int, int]:
    passed = 0
    for s in samples:
        pred = s["filtered_resps"][0][0]
        target = s["target"]
        passed += int(run_py(pred + "\n" + target))
    return passed / len(samples), passed, len(samples)


def eval_mbpp(samples: list) -> tuple[float, int, int]:
    passed = 0
    for s in samples:
        pred = s["filtered_resps"]
        if isinstance(pred, list):
            pred = pred[0]
        setup = s["doc"].get("test_setup_code") or ""
        target = s["target"]
        program = (setup + "\n" if setup else "") + pred + "\n" + target
        passed += int(run_py(program))
    return passed / len(samples), passed, len(samples)


with open(RESULTS_PATH) as f:
    data = json.load(f)

all_samples = data["tasks"]["samples"]

results = {}
for task, eval_fn in [("humaneval", eval_humaneval), ("mbpp", eval_mbpp)]:
    if task in all_samples:
        rate, passed, total = eval_fn(all_samples[task])
        results[task] = (rate, passed, total)
        print(f"{task:12s}  pass@1 = {rate:.3f}  ({passed}/{total})")
    else:
        print(f"{task:12s}  not found in results.json")
