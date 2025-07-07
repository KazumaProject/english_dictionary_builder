# (0) ライブラリのインポート
from datasets import load_dataset
from collections import Counter
import numpy as np
from tqdm.auto import tqdm
import nltk
from nltk.util import ngrams

# --------------------------------------------------------------------------
# ★★★ 設定項目 ★★★
# 1. 解析したいN-gramのサイズを指定 (1: unigram, 2: bigram, 3: trigram)
N_GRAM_SIZE = 2

# 2. スコアの種類を選択 ('cost' または 'pmi')
#   'cost': 高精度コスト（対数確率）。N-gramの珍しさを表す。
#   'pmi':  自己相互情報量。単語間の関連性の強さを表す (N_GRAM_SIZE >= 2 の場合のみ有効)。
SCORE_TYPE = 'pmi'
# --------------------------------------------------------------------------

# --- 設定チェック ---
if N_GRAM_SIZE == 1 and SCORE_TYPE == 'pmi':
    print("Warning: PMI is for N-grams of size 2 or more. Switching SCORE_TYPE to 'cost'.")
    SCORE_TYPE = 'cost'

# 出力ファイル名を動的に設定
output_filename = f"{N_GRAM_SIZE}-grams_score_{SCORE_TYPE}.txt"

# NLTKのデータ準備
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# (1) データセットのメタ情報読み込み
print("Loading dataset metadata...")
train_ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="train")
total_docs = len(train_ds)

# (2) N-gramの頻度を計算
print(f"Starting to count unigrams and {N_GRAM_SIZE}-grams...")
dataset = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="train", streaming=True)

unigram_counts = Counter()
ngram_counts = Counter()

for item in tqdm(dataset, total=total_docs, desc="Docs processed"):
    text = item["text"].lower()
    tokens = nltk.word_tokenize(text)
    words = [word for word in tokens if word.isalpha()]
    
    # PMI計算のためにユニグラムは常にカウント
    unigram_counts.update(words)
    
    if len(words) >= N_GRAM_SIZE:
        generated_ngrams = ngrams(words, N_GRAM_SIZE)
        ngram_counts.update(generated_ngrams)

print("✅ Frequency counting complete.")

# (3) 指定された種類のスコアを計算
print(f"Calculating '{SCORE_TYPE}' scores...")

total_unigrams = sum(unigram_counts.values())
total_ngrams = sum(ngram_counts.values())
results_data = []

if SCORE_TYPE == 'cost':
    for ngram_tuple, count in tqdm(ngram_counts.items(), desc="Calculating costs"):
        freq = count / total_ngrams
        if freq > 0:
            # より精密な対数確率スコア（大きいほど珍しい）
            score = -np.log(freq)
            results_data.append((" ".join(ngram_tuple), score))
    # スコアが高い順（珍しい順）にソート
    results_data.sort(key=lambda x: x[1], reverse=True)

elif SCORE_TYPE == 'pmi':
    # PMIは頻度が低いと値が不安定になるため、最低出現回数を設定
    MIN_FREQ_THRESHOLD = 5
    
    for ngram_tuple, count in tqdm(ngram_counts.items(), desc="Calculating PMI"):
        if count < MIN_FREQ_THRESHOLD:
            continue
            
        p_ngram = count / total_ngrams
        
        # 個々の単語の出現確率を計算
        p_words_independent = 1.0
        for word in ngram_tuple:
            p_words_independent *= unigram_counts[word] / total_unigrams
        
        if p_ngram > 0 and p_words_independent > 0:
            # PMIを計算 (log2を使用するのが一般的)
            pmi_score = np.log2(p_ngram / p_words_independent)
            results_data.append((" ".join(ngram_tuple), pmi_score))
    # PMIスコアが高い順（関連性が強い順）にソート
    results_data.sort(key=lambda x: x[1], reverse=True)


# (4) 結果をファイルに保存
with open(output_filename, "w", encoding="utf-8") as f:
    for ngram_string, score in results_data:
        # スコアを小数点第4位までフォーマット
        f.write(f"{ngram_string}\t{score:.4f}\n")

print(f"✅ Done! Saved scores to '{output_filename}'")
