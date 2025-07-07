# (0) ライブラリのインポート
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
required_nltk_packages = ['punkt', 'words']
for package in required_nltk_packages:
    try:
        nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else f'corpora/{package}')
    except LookupError:
        print(f"Downloading NLTK package: {package}")
        nltk.download(package)

# (1) 英単語辞書をセットとして読み込み
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
    words = [word for word in tokens if word.isalpha() and word in english_words]
    
    if N_GRAM_SIZE == 1:
        ngram_counts.update(words)
    else:
        unigram_counts.update(words)
        if len(words) >= N_GRAM_SIZE:
            ngram_counts.update(ngrams(words, N_GRAM_SIZE))
print("✅ Frequency counting complete.")


# (4) スコアを計算 (浮動小数点)
print(f"Calculating floating point '{SCORE_TYPE}' scores...")
float_scores_data = []

if SCORE_TYPE == 'pmi':
    total_unigrams = sum(unigram_counts.values())
    total_ngrams = sum(ngram_counts.values())
    for ngram_tuple, count in tqdm(ngram_counts.items(), desc="Calculating PMI"):
        p_ngram = count / total_ngrams
        p_words_independent = np.prod([unigram_counts[word] / total_unigrams for word in ngram_tuple])
        if p_ngram > 0 and p_words_independent > 0:
            float_scores_data.append((" ".join(ngram_tuple), np.log2(p_ngram / p_words_independent)))
else: # 'cost'
    total_ngrams = sum(ngram_counts.values())
    for ngram_tuple, count in tqdm(ngram_counts.items(), desc="Calculating costs"):
        if count > 0:
            ngram_str = " ".join(ngram_tuple) if isinstance(ngram_tuple, tuple) else ngram_tuple
            float_scores_data.append((ngram_str, -np.log(count / total_ngrams)))

# (5) ★★★ スコアを0-65535の範囲に正規化 ★★★
print("Normalizing scores to short integer range (0-65535)...")
if float_scores_data:
    # 元のスコアを抽出
    scores = [item[1] for item in float_scores_data]
    min_score, max_score = min(scores), max(scores)
    
    final_data = []
    score_range = max_score - min_score
    if score_range == 0: score_range = 1 # ゼロ除算を防止
    
    for ngram_string, score in float_scores_data:
        if SCORE_TYPE == 'pmi':
            # PMIは値が高いほど良い -> スコアを反転させて低い値(0に近い)にする
            scaled_score = 65535 - int(((score - min_score) / score_range) * 65535)
        else: # cost
            # costは値が低いほど良い -> そのまま低い値にする
            scaled_score = int(((score - min_score) / score_range) * 65535)
        final_data.append((ngram_string, scaled_score))
else:
    final_data = []
    
# (6) ★★★ 最終スコアが低い順にソート ★★★
final_data.sort(key=lambda x: x[1])

# (7) 結果をファイルに保存
with open(output_filename, "w", encoding="utf-8") as f:
    for ngram_string, score in final_data:
        f.write(f"{ngram_string}\t{score}\n")

print(f"✅ Done! Saved scores to '{output_filename}'")
