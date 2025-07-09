#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge.py
--------
counts_part*.txt を統合してコストスコアを計算し、
1-grams_score_cost_pos_combined.txt を生成する。
"""
import glob, math
from collections import Counter

OUTPUT = "1-grams_score_cost_pos_combined.txt"
counts_files = sorted(glob.glob("counts_part*.txt"))
assert counts_files, "counts_part*.txt が見付かりません"

unigram_counts: Counter[str] = Counter()
word_details: dict[str, dict[str, str]] = {}

for path in counts_files:
    with open(path, encoding="utf-8") as f:
        for line in f:
            lower, original, pos, cnt = line.rstrip("\n").split("\t")
            cnt = int(cnt)
            unigram_counts[lower] += cnt
            if lower not in word_details or pos == "PROPN":
                word_details[lower] = {"original": original, "pos": pos}

total = sum(unigram_counts.values())
min_cost = math.inf
max_cost = -math.inf
float_scores = []

for w, c in unigram_counts.items():
    cost = -math.log(c / total)
    min_cost = min(min_cost, cost)
    max_cost = max(max_cost, cost)
    d = word_details[w]
    float_scores.append({"lower": w, "original": d["original"], "pos": d["pos"], "cost": cost})

scale = max_cost - min_cost or 1  # 0 で割らない
for x in float_scores:
    x["scaled"] = int(((x["cost"] - min_cost) / scale) * 65535)

float_scores.sort(key=lambda x: x["scaled"])

with open(OUTPUT, "w", encoding="utf-8") as f:
    f.write("input_word\toutput_word\tpos_tag\tscore\n")
    for x in float_scores:
        f.write(f"{x['lower']}\t{x['original']}\t{x['pos']}\t{x['scaled']}\n")

print(f"✅ Wrote {OUTPUT}")
