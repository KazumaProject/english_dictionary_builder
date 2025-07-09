#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
worker.py
---------
• PART       : 0-indexed ジョブ番号（GitHub Actions の matrix で渡す）
• NUM_PARTS  : 総ジョブ数
各ジョブが Wikipedia ストリームのうち (index % NUM_PARTS == PART) の記事だけを解析し、
└ counts_part{PART}.txt を出力する。
"""
import os
from collections import Counter
from datasets import load_dataset
from tqdm.auto import tqdm
import spacy

PART      = int(os.getenv("PART", "0"))
NUM_PARTS = int(os.getenv("NUM_PARTS", "1"))

# ───────── spaCy ──────────
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
nlp.max_length = 2_000_000  # 2 MB まで許容（必要ならさらに拡大）

# ───────── データセット定義 ──────────
DATASET_CONFIGS = [
    {"name": "wikipedia", "config": "20220301.en", "split": "train", "column": "text"},
]

unigram_counts: Counter[str] = Counter()
word_details: dict[str, dict[str, str]] = {}

for cfg in DATASET_CONFIGS:
    ds = load_dataset(cfg["name"], cfg["config"], split=cfg["split"],
                      streaming=True, trust_remote_code=True)

    # (index % NUM_PARTS == PART) の記事のみ処理
    texts_iter = (
        chunk
        for idx, item in enumerate(ds)
        if idx % NUM_PARTS == PART
        for chunk in item[cfg["column"]].split("\n\n")  # 段落で分割
        if chunk.strip()
    )

    for doc in tqdm(nlp.pipe(texts_iter, batch_size=256),
                    desc=f"PART {PART}", unit="doc"):
        tokens = []
        for t in doc:
            if t.is_alpha and not t.is_stop:
                lw = t.lower_
                tokens.append(lw)
                if lw not in word_details or t.pos_ == "PROPN":
                    word_details[lw] = {"original": t.text, "pos": t.pos_}
        unigram_counts.update(tokens)

# ───────── 部分結果を書き出し ──────────
outfile = f"counts_part{PART}.txt"
with open(outfile, "w", encoding="utf-8") as f:
    for w, c in unigram_counts.items():
        d = word_details[w]
        f.write(f"{w}\t{d['original']}\t{d['pos']}\t{c}\n")
print(f"✅ Saved {outfile}")
