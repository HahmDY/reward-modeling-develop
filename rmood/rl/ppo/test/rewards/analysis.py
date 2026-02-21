import json
import os
import re
import numpy as np
from collections import defaultdict

BASE = "/root/reward-modeling-develop/rmood/rl/ppo/test/rewards"

def load_values(path):
    with open(path) as f:
        data = json.load(f)
    return [item[0] for item in data]

def parse_exp_name(dirname):
    """디렉토리 이름에서 모델명과 step 추출"""
    match = re.search(r"(.+)-ppo-step-(\d+)$", dirname)
    if match:
        return match.group(1), int(match.group(2))
    return dirname, -1

# 데이터 수집
results = defaultdict(dict)  # results[model][step][rm_name] = values

for dirname in sorted(os.listdir(BASE)):
    dirpath = os.path.join(BASE, dirname)
    if not os.path.isdir(dirpath):
        continue
    model, step = parse_exp_name(dirname)
    for fname in os.listdir(dirpath):
        if not fname.endswith(".json"):
            continue
        rm_name = fname.replace(".json", "")
        values = load_values(os.path.join(dirpath, fname))
        results[model][step] = results[model].get(step, {})
        results[model][step][rm_name] = values

# ── 1. 전체 요약 테이블 ───────────────────────────────────────────
print("\n" + "=" * 90)
print("[ 전체 평균 요약 ]")
print("=" * 90)
print("%-48s %5s  %-38s %8s %8s %8s" % ("모델", "step", "평가 RM", "평균", "std", "min/max"))
print("-" * 90)

for model in sorted(results):
    for step in sorted(results[model]):
        for rm, vals in sorted(results[model][step].items()):
            arr = np.array(vals)
            print("%-48s %5d  %-38s %8.4f %8.4f %4.1f/%4.1f" % (
                model, step, rm, arr.mean(), arr.std(), arr.min(), arr.max()))
    print()

# ── 2. 모델별 step 추이 (Skywork 기준) ───────────────────────────
print("=" * 70)
print("[ Skywork 리워드 기준 step별 추이 ]")
print("=" * 70)
skywork_key = "Skywork-Reward-V2-Llama-3.1-8B"

for model in sorted(results):
    print(f"\n  {model}")
    print("  %6s  %8s  %8s  %8s" % ("step", "mean", "std", "min/max"))
    print("  " + "-" * 40)
    for step in sorted(results[model]):
        if skywork_key in results[model][step]:
            arr = np.array(results[model][step][skywork_key])
            print("  %6d  %8.4f  %8.4f  %4.1f/%5.1f" % (
                step, arr.mean(), arr.std(), arr.min(), arr.max()))

# ── 3. 모델별 step 추이 (자체 RM 기준) ──────────────────────────
print("\n" + "=" * 70)
print("[ 자체 RM 기준 step별 추이 ]")
print("=" * 70)

for model in sorted(results):
    own_keys = [k for k in next(iter(results[model].values())) if "Skywork" not in k]
    for own_key in own_keys:
        print(f"\n  {model}  /  {own_key}")
        print("  %6s  %8s  %8s  %8s" % ("step", "mean", "std", "min/max"))
        print("  " + "-" * 40)
        for step in sorted(results[model]):
            if own_key in results[model][step]:
                arr = np.array(results[model][step][own_key])
                print("  %6d  %8.4f  %8.4f  %4.1f/%5.1f" % (
                    step, arr.mean(), arr.std(), arr.min(), arr.max()))

# ── 4. Skywork 기준 모델 간 비교 ─────────────────────────────────
print("\n" + "=" * 70)
print("[ 모델 간 Skywork 비교 (공통 step) ]")
print("=" * 70)
all_steps = set()
for model in results:
    all_steps |= set(results[model].keys())

models = sorted(results.keys())
header = "%6s" % "step"
for m in models:
    header += "  %-20s" % m[-15:]
print(header)
print("-" * 70)
for step in sorted(all_steps):
    row = "%6d" % step
    for m in models:
        if step in results[m] and skywork_key in results[m][step]:
            row += "  %20.4f" % np.mean(results[m][step][skywork_key])
        else:
            row += "  %20s" % "N/A"
    print(row)