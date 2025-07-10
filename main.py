# (0) 追加ライブラリ
from datasets import load_dataset, get_dataset_config_names
from collections import Counter
import numpy as np
from tqdm.auto import tqdm
import spacy
from itertools import islice
import re

# ----------------------------------------------------------------------
# ★★★ 設定項目 ★★★
N_GRAM_SIZE = 1
SCORE_TYPE   = "cost"

# GitHub Actions の 6h 制限内で収まるように調整
MAX_SAMPLES_TO_PROCESS = 750_000     # Wikipedia は 1 記事が長いので少し減らす

# ── (NEW) 最新 Wikipedia 英語ダンプの config 名を自動取得 ──
all_cfgs         = get_dataset_config_names("wikimedia/wikipedia")
LATEST_WIKI_CFG  = max(c for c in all_cfgs if re.match(r"\d{8}\.en$", c))   # yyyyMMdd.en

DATASET_CONFIGS = [
    # 最新 Wikipedia 英語
    {"name": "wikimedia/wikipedia", "config": LATEST_WIKI_CFG, "split": "train", "column": "text"},
    # 定番のオープンデータセット（小さめで動作確認しやすい）
    {"name": "wikitext", "config": "wikitext-103-v1", "split": "train", "column": "text"},
]

output_filename = f"{N_GRAM_SIZE}-grams_{SCORE_TYPE}_{LATEST_WIKI_CFG}_wikitext.txt"
# ----------------------------------------------------------------------

# (1) spaCy モデル読み込み（高速化用に parser/ner 無効化）
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
print("✅ spaCy model loaded.")

# (2) カウンターと単語メタ
unigram_counts = Counter()
word_details   = {}

# (3) データセットを順番に処理
for cfg in DATASET_CONFIGS:
    print(f"\n📚 Processing {cfg['name']} ({cfg['config']}) …")
    ds_stream = load_dataset(
        path   = cfg["name"],
        name   = cfg.get("config"),
        split  = cfg["split"],
        streaming = True,
        # Wikipedia は Apache-Beam が必要な場合がある
        beam_runner = "DirectRunner",
        trust_remote_code = True,
    )

    # 必要件数だけ取り出す
    ds_subset = islice(ds_stream, MAX_SAMPLES_TO_PROCESS)
    texts     = (row[cfg["column"]] for row in ds_subset)

    for doc in tqdm(nlp.pipe(texts, batch_size=500), total=MAX_SAMPLES_TO_PROCESS,
                    desc=f"spaCy ↔ {cfg['name']}"):
        toks = [t.lower_ for t in doc if t.is_alpha and not t.is_stop]
        for t in toks:
            if t not in word_details or doc[toks.index(t)].pos_ == "PROPN":
                word_details[t] = {"original": t, "pos": doc[toks.index(t)].pos_}
        unigram_counts.update(toks)

print("\n✅ すべてのデータセットの集計完了")

# ── 以下 (4)〜(7) はほぼそのまま ──
# ...
