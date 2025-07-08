#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate 1-gram cost scores and extract proper nouns.

Outputs
  - 1-grams_score_cost_pos_combined.txt
  - proper_nouns.txt
Budget is controlled by environment variables:
  TIME_BUDGET_MIN (minutes)   – default 240
  TOKEN_BUDGET    (tokens)    – default 0 (no limit)
"""

import os
import time
import re
from collections import Counter, defaultdict
from itertools import islice

import spacy
import numpy as np
from datasets import load_dataset
from tqdm.auto import tqdm

# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------
N_GRAM_SIZE = 1
SCORE_TYPE = "cost"
OUTPUT_FREQ_FILE = f"{N_GRAM_SIZE}-grams_score_{SCORE_TYPE}_pos_combined.txt"
OUTPUT_PROPN_FILE = "proper_nouns.txt"

# Data sources – 好みに応じて追加/削除
DATASET_CONFIGS = [
    # Smaller sanity-check corpus
    {
        "name": "wikitext",
        "config": "wikitext-103-v1",
        "split": "train",
        "column": "text",
    },
    # Full English Wikipedia dump (Jan-2025). 最新版に置き換えても可
    {
        "name": "wikimedia/wikipedia",
        "config": "20250101.en",
        "split": "train",
        "column": "text",
    },
]

# Budgets (env overrides)
TIME_BUDGET_MIN = int(os.getenv("TIME_BUDGET_MIN", "240"))     # minutes
TOKEN_BUDGET = int(os.getenv("TOKEN_BUDGET", "0"))             # 0 = unlimited

# Batch & parallelism
BATCH_SIZE = 1000          # ↑ RAM が許せば 2000 などへ
N_PROCESS = 2              # GitHub Standard Runner は 2 vCPU

# ---------------------------------------------------------------------
# Helper – Proper-noun heuristic
# ---------------------------------------------------------------------
_CAMEL_RE = re.compile(r"[A-Z]")

def looks_like_propn(tok) -> bool:
    """
    Heuristic:
      * alphabetic
      * either starts with upper-case inside sentence OR contains inner capitals
        (e.g. iPhone, eBay, McDonald’s)
    """
    if not tok.is_alpha:
        return False
    # Exclude very first token of each streamed doc (often sentence start)
    if tok.i == 0:
        return False
    txt = tok.text
    return txt[0].isupper() or bool(_CAMEL_RE.search(txt[1:]))

# ---------------------------------------------------------------------
# Load spaCy – tagger のみ（parser/ner は無効化）
# ---------------------------------------------------------------------
print("📦 Loading spaCy model …")
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
print("✅ spaCy model loaded.")

# ---------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------
unigram_counts: Counter[str] = Counter()
propn_counts:   Counter[str] = Counter()
word_meta: dict[str, dict[str, str]] = {}      # lower -> {original,pos}

# ---------------------------------------------------------------------
# Iterate datasets with budget checks
# ---------------------------------------------------------------------
start_time = time.perf_counter()
tokens_seen = 0

for cfg in DATASET_CONFIGS:
    print(f"\n🔄 Processing {cfg['name']} ({cfg['split']}) …")
    ds = load_dataset(
        cfg["name"],
        cfg.get("config"),
        split=cfg["split"],
        streaming=True,
        trust_remote_code=True,
    )
    texts = (item[cfg["column"]] for item in ds)

    for doc in tqdm(
        nlp.pipe(texts, batch_size=BATCH_SIZE, n_process=N_PROCESS),
        desc=f"spaCy ({cfg['name']})",
    ):
        tokens_seen += len(doc)

        batch_words = []
        for tok in doc:
            if tok.is_alpha and not tok.is_stop:
                lower = tok.lower_
                batch_words.append(lower)

                # meta
                if lower not in word_meta or tok.pos_ == "PROPN":
                    word_meta[lower] = {"original": tok.text, "pos": tok.pos_}

                # proper-noun collection
                if tok.pos_ == "PROPN" or looks_like_propn(tok):
                    propn_counts[lower] += 1

        unigram_counts.update(batch_words)

        # Budget check
        elapsed_min = (time.perf_counter() - start_time) / 60
        if (TIME_BUDGET_MIN and elapsed_min >= TIME_BUDGET_MIN) or \
           (TOKEN_BUDGET and tokens_seen >= TOKEN_BUDGET):
            print(f"⏹️  Budget reached "
                  f"({elapsed_min:.1f} min / {tokens_seen:_} tok). Stopping.")
            break
    else:
        # for-loop finished naturally → continue next dataset
        continue
    break   # inner break → exit outer loop

print("\n✅ Counting finished.")

# ---------------------------------------------------------------------
# Cost score → 0–65535
# ---------------------------------------------------------------------
print("🧮 Calculating scores …")
total = sum(unigram_counts.values())
float_scores = []
for w, cnt in unigram_counts.items():
    if w not in word_meta or cnt == 0:
        continue
    cost = -np.log(cnt / total)
    float_scores.append((w, cost))

if not float_scores:
    raise RuntimeError("No data collected – check budgets / filters.")

scores_only = [c for _, c in float_scores]
min_s, max_s = min(scores_only), max(scores_only)
rng = max(max_s - min_s, 1e-9)
scale = lambda x: int(((x - min_s) / rng) * 65535)

# ---------------------------------------------------------------------
# Write files
# ---------------------------------------------------------------------
print(f"💾 Writing {OUTPUT_FREQ_FILE} …")
with open(OUTPUT_FREQ_FILE, "w", encoding="utf-8") as f:
    f.write("input_word\toutput_word\tpos_tag\tscore\n")
    for w, s in sorted(float_scores, key=lambda t: scale(t[1])):
        meta = word_meta[w]
        f.write(f"{w}\t{meta['original']}\t{meta['pos']}\t{scale(s)}\n")

print(f"💾 Writing {OUTPUT_PROPN_FILE} …")
with open(OUTPUT_PROPN_FILE, "w", encoding="utf-8") as f:
    f.write("proper_noun\tcount\n")
    for w, c in propn_counts.most_common():
        f.write(f"{w}\t{c}\n")

print("🎉 Done.")
