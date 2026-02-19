import json

def check_correctness(completion, test):
    try:
        exec(completion + "\n" + test, {})
        return True
    except Exception:
        return False

with open("results.json") as f:
    data = json.load(f)

samples = data["tasks"]["samples"]["humaneval"]
passed = 0
for s in samples:
    pred = s["filtered_resps"][0][0]
    target = s["target"]
    if check_correctness(pred, target):
        passed += 1

print(f"pass@1 = {passed / len(samples):.3f}  ({passed}/{len(samples)})")
