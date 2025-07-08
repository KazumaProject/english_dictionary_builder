#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-speed n-gram scorer
 - spaCy tagger/lemmatizer を無効化
 - マルチプロセス (n_process > 1)
 - PROPN 判定は簡易ヒューリスティック
"""

from collections import Counter
from itertools import islice
import math
import pathlib
import sys

import numpy as np
from datasets import load_dataset
from tqdm.auto import tqdm
import spacy
from spacy.attrs import LOWER, IS_ALPHA, IS_STOP

# --------------------------------------------------------------------------
# ★★★ 設定項目 ★★★
N_GRAM_SIZE = 1
SCORE_TYPE = "cost"          # 現状 'cost' 固定
SAMPLE_RATE = 1.0            # 0.1 → 10 % サンプリング
BATCH_SIZE = 4000            # 1 プロセスあたりのバッチ
N_PROCESS = 4                # CPU コア数に合わせて変更
DATASET_CONFIGS = [
    {
        "name": "wikitext",
        "config": "wikitext-103-v1",
        "split": "train",
        "column": "text",
    },
]
# --------------------------------------------------------------------------

output_filename = f"{N_GRAM_SIZE}-grams_score_{SCORE_TYPE}_pos_combined.txt"

# (0) spaCy 初期化（Tokenizer のみ）
print("Loading spaCy tokenizer …")
nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser",
                                            "lemmatizer", "ner"])
print("✅ tokenizer loaded.")

# (1) カウンターとメタ情報
unigram_counts: Counter[str] = Counter()
word_details: dict[str, dict[str, str]] = {}

# (2) データセットごとに処理
for cfg in DATASET_CONFIGS:
    print(f"\nProcessing dataset: {cfg['name']} / {cfg['split']}")
    ds = load_dataset(
        cfg["name"],
        cfg.get("config"),
        split=cfg["split"],
        streaming=True,
        trust_remote_code=True,
    )

    # ストリーミング Dataset → サンプリングしつつジェネレータ化
    def _texts():
        for idx, item in enumerate(ds):
            if SAMPLE_RATE < 1.0 and (idx % int(1 / SAMPLE_RATE)):
                continue
            yield item[cfg["column"]]

    texts_generator = _texts()

    # spaCy をマルチプロセスで回す
    for doc in tqdm(
        nlp.pipe(texts_generator,
                 batch_size=BATCH_SIZE,
                 n_process=N_PROCESS),
        desc=f"spaCy ({cfg['name']})",
    ):
        # ndarray ベースで高速抽出
        arr = doc.to_array([LOWER, IS_ALPHA, IS_STOP])
        lowers = []
        for lower_id, is_alpha, is_stop in arr:
            if is_alpha and not is_stop:
                lowers.append(lower_id)

        # カウンタ更新
        unigram_counts.update(lowers)

        # PROPN 相当かどうか（先頭大文字ヒューリスティック）
        for token in doc:
            lw = token.lower_
            if lw not in word_details:
                is_prop = token.text[:1].isupper()
                word_details[lw] = {
                    "original": token.text,
                    "pos": "PROPN" if is_prop else "X",
                }

# (3) スコア計算
print("\nCalculating scores …")
total = sum(unigram_counts.values())
float_scores = {
    lw: -math.log(c / total) for lw, c in unigram_counts.items() if c > 0
}

# (4) 0–65535 スケールに正規化
min_s, max_s = min(float_scores.values()), max(float_scores.values())
rng = max_s - min_s or 1.0
scaled = {
    lw: int(((s - min_s) / rng) * 65535) for lw, s in float_scores.items()
}

# (5) 出力
print(f"Writing → {output_filename}")
with open(output_filename, "w", encoding="utf-8") as f:
    f.write("input_word\toutput_word\tpos_tag\tscore\n")
    for lw, score in sorted(scaled.items(), key=lambda x: x[1]):
        meta = word_details.get(lw, {"original": lw, "pos": "X"})
        f.write(f"{lw}\t{meta['original']}\t{meta['pos']}\t{score}\n")

print("✅ Done!")
