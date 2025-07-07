# (0) ライブラリのインポート
import os
from datasets import load_dataset
from collections import Counter
import numpy as np
from tqdm.auto import tqdm
import nltk
from nltk.util import ngrams
from nltk.corpus import words as nltk_words

# --------------------------------------------------------------------------
# ★★★ 設定項目 ★★★
N_GRAM_SIZE = 1
SCORE_TYPE = 'cost'
# --------------------------------------------------------------------------

# --- 設定チェックとファイル名設定 ---
if N_GRAM_SIZE == 1 and SCORE_TYPE == 'pmi':
    print("Warning: PMI is for N-grams of size 2 or more. Switching to 'cost'.")
    SCORE_TYPE = 'cost'
output_filename = f"{N_GRAM_SIZE}-grams_score_{SCORE_TYPE}.txt"

# --- NLTKデータ準備 ---
# 必要なNLTKパッケージのリスト
required_nltk_packages = ['punkt', 'words']
for package in required_nltk_packages:
    try:
        nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else f'corpora/{package}')
    except LookupError:
        print(f"Downloading NLTK package: {package}")
        nltk.download(package)

# (1) 英単語辞書をセットとして読み込み、高速な検索を可能にする
print("Loading English dictionary...")
english_words = set(nltk_words.words())
print(f"✅ Loaded {len(english_words)} words from English dictionary.")


# (2) データセットのメタ情報読み込み
print("Loading dataset metadata...")
train_ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="train")
total_docs = len(train_ds)

# (3) N-gramの頻度を計算
print(f"Starting to count {N_GRAM_SIZE}-grams...")
dataset = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="train", streaming=True)

unigram_counts = Counter()
ngram_counts = Counter()

for item in tqdm(dataset, total=total_docs, desc="Docs processed"):
    text = item["text"].lower()
    tokens = nltk.word_tokenize(text)
    
    # ★★★ 改善点: 英単語辞書でフィルタリング ★★★
    # isalpha()チェックに加え、辞書に存在する単語のみを対象とする
    words = [word for word in tokens if word.isalpha() and word in english_words]

    if N_GRAM_SIZE == 1:
        ngram_counts.update(words)
    else:
        # PMI計算のためにユニグラムは常にカウント
        unigram_counts.update(words)
        if len(words) >= N_GRAM_SIZE:
            generated_ngrams = ngrams(words, N_GRAM_SIZE)
            ngram_counts.update(generated_ngrams)

print("✅ Frequency counting complete.")

# (4) スコアを計算
print(f"Calculating '{SCORE_TYPE}' scores...")
results_data = []

if N_GRAM_SIZE > 1 and SCORE_TYPE == 'pmi':
    total_unigrams = sum(unigram_counts.values())
    total_ngrams = sum(ngram_counts.values())
    MIN_FREQ_THRESHOLD = 5
    for ngram_tuple, count in tqdm(ngram_counts.items(), desc="Calculating PMI"):
        if count < MIN_FREQ_THRESHOLD: continue
        p_ngram = count / total_ngrams
        p_words_independent = np.prod([unigram_counts[word] / total_unigrams for word in ngram_tuple])
        if p_ngram > 0 and p_words_independent > 0:
            pmi_score = np.log2(p_ngram / p_words_independent)
            results_data.append((" ".join(ngram_tuple), pmi_score))
    results_data.sort(key=lambda x: x[1], reverse=True)
else: # 'cost' スコアの計算
    total_ngrams = sum(ngram_counts.values())
    for ngram_tuple, count in tqdm(ngram_counts.items(), desc="Calculating costs"):
        freq = count / total_ngrams
        if freq > 0:
            score = -np.log(freq)
            # N-gramがタプルで来る場合と文字列で来る場合に対応
            ngram_str = " ".join(ngram_tuple) if isinstance(ngram_tuple, tuple) else ngram_tuple
            results_data.append((ngram_str, score))
    results_data.sort(key=lambda x: x[1], reverse=True)

# (5) 結果をファイルに保存
with open(output_filename, "w", encoding="utf-8") as f:
    for ngram_string, score in results_data:
        f.write(f"{ngram_string}\t{score:.4f}\n")

print(f"✅ Done! Saved scores to '{output_filename}'")
