# wikipedia_ngram.py
from pathlib import Path
import json, spacy, numpy as np
from collections import Counter
from tqdm.auto import tqdm

# ---- 設定 -----------------------------------------------------
N_GRAM_SIZE   = 1              # 2,3... にしても OK
SCORE_TYPE    = "cost"
OUTPUT_FILE   = f"{N_GRAM_SIZE}-gram_{SCORE_TYPE}_enwiki.txt"
EXTRACT_DIR   = Path("wiki_json")   # 2️⃣ で指定した出力先
BATCH_SIZE    = 500            # RAM に合わせて調整
# --------------------------------------------------------------

print("Loading spaCy...")
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
nlp.max_length = 2_000_000      # 長い記事対策
print("spaCy ready ✔")

counter, meta = Counter(), {}
files = list(EXTRACT_DIR.rglob("wiki_*"))  # ≈ 30k ファイル
print(f"{len(files):,} part-files found – start processing")

def iter_articles():
    for fp in files:
        with fp.open(encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)["text"]

docs = nlp.pipe(iter_articles(), batch_size=BATCH_SIZE)
for doc in tqdm(docs, total=None, desc="spaCy"):
    words = [t.lower_ for t in doc if t.is_alpha and not t.is_stop]
    for t in words:
        if t not in meta or doc[t.i].pos_ == "PROPN":
            meta[t] = {"orig": t, "pos": doc[t.i].pos_}
    counter.update(words)

print("Counting finished – computing scores")
tot = sum(counter.values())
scores = {w: -np.log(c / tot) for w, c in counter.items()}

mn, mx = min(scores.values()), max(scores.values())
rng = mx - mn or 1
scaled = {w: int((s - mn) / rng * 65535) for w, s in scores.items()}

print(f"Writing {OUTPUT_FILE} …")
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("input_word\toutput_word\tpos_tag\tscore\n")
    for w, sc in sorted(scaled.items(), key=lambda x: x[1]):
        info = meta[w]
        f.write(f"{w}\t{info['orig']}\t{info['pos']}\t{sc}\n")

print("✅ Done!")
