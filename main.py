# (0) ライブラリのインポート
from datasets import load_dataset, get_dataset_infos
from collections import Counter
import numpy as np
from tqdm.auto import tqdm
import nltk
from nltk.util import ngrams
from nltk.corpus import words as nltk_words

# --------------------------------------------------------------------------
# ★★★ 設定項目 ★★★
# 1. 解析したいN-gramのサイズを指定 (1: unigram, 2: bigram, 3: trigram)
N_GRAM_SIZE = 1

# 2. スコアの種類を選択 ('cost' または 'pmi')
SCORE_TYPE = 'cost'

# 3. ★★★ 使用するデータセットのリスト ★★★
DATASET_CONFIGS = [
    {"name": "wikipedia", "config": "20220301.en", "split": "train", "column": "text"},
    # {"name": "bookcorpus", "config": None, "split": "train", "column": "text"},
    # C4データセットの一部（最初の1%）を追加。'1%'を'5%'や'100000'（行数）などに変更可能
    # {"name": "c4", "config": "en", "split": "train[:200000]", "column": "text"},
]
# --------------------------------------------------------------------------

# --- ファイル名設定 ---
output_filename = f"{N_GRAM_SIZE}-grams_score_{SCORE_TYPE}_combined.txt"

# --- NLTKデータ準備 ---
required_nltk_packages = ['punkt', 'words']
for package in required_nltk_packages:
    try:
        nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else f'corpora/{package}')
    except LookupError:
        nltk.download(package)

# (1) 英単語辞書をセットとして読み込み
print("Loading English dictionary...")
english_words = set(nltk_words.words())
print(f"✅ Loaded {len(english_words)} words from English dictionary.")


# (2) カウンターを初期化
unigram_counts = Counter()
ngram_counts = Counter()

# (3) ★★★ 複数のデータセットを順番に処理 ★★★
for config in DATASET_CONFIGS:
    dataset_name = config["name"]
    print(f"\nProcessing dataset: {dataset_name} (split: {config['split']})...")

    # データセットの総ドキュメント数を取得 (プログレスバー用)
    # スライスを使うと正確な合計が取れない場合があるため、totalをNoneに設定
    total_docs = None

    # データセットをストリーミングで読み込み
    dataset = load_dataset(
        dataset_name,
        config.get("config"),
        split=config["split"],
        streaming=True,
        trust_remote_code=True
    )

    # N-gramの頻度を計算
    for item in tqdm(dataset, total=total_docs, desc=f"Processing {dataset_name}"):
        text = item[config["column"]].lower()
        tokens = nltk.word_tokenize(text)
        
        words = [word for word in tokens if word.isalpha() and word in english_words]

        if N_GRAM_SIZE == 1:
            ngram_counts.update(words)
        else:
            unigram_counts.update(words)
            if len(words) >= N_GRAM_SIZE:
                ngram_counts.update(ngrams(words, N_GRAM_SIZE))

print("\n✅ All datasets processed. Frequency counting complete.")


# (4) スコアを計算 (浮動小数点)
print(f"Calculating floating point '{SCORE_TYPE}' scores...")
float_scores_data = []

if SCORE_TYPE == 'pmi' and N_GRAM_SIZE > 1:
    total_unigrams = sum(unigram_counts.values())
    total_ngrams = sum(ngram_counts.values())
    for ngram_tuple, count in tqdm(ngram_counts.items(), desc="Calculating PMI"):
        p_ngram = count / total_ngrams
        p_words_independent = np.prod([unigram_counts.get(word, 0) / total_unigrams for word in ngram_tuple])
        if p_ngram > 0 and p_words_independent > 0:
            float_scores_data.append((" ".join(ngram_tuple), np.log2(p_ngram / p_words_independent)))
else: # 'cost'
    total_ngrams = sum(ngram_counts.values())
    for ngram_tuple, count in tqdm(ngram_counts.items(), desc="Calculating costs"):
        if count > 0:
            ngram_str = " ".join(ngram_tuple) if isinstance(ngram_tuple, tuple) else ngram_tuple
            float_scores_data.append((ngram_str, -np.log(count / total_ngrams)))

# (5) スコアを0-65535の範囲に正規化
print("Normalizing scores to short integer range (0-65535)...")
if float_scores_data:
    scores = [item[1] for item in float_scores_data]
    min_score, max_score = min(scores), max(scores)
    score_range = max_score - min_score
    if score_range == 0: score_range = 1

    final_data = []
    for ngram_string, score in float_scores_data:
        if SCORE_TYPE == 'pmi':
            scaled_score = 65535 - int(((score - min_score) / score_range) * 65535)
        else:
            scaled_score = int(((score - min_score) / score_range) * 65535)
        final_data.append((ngram_string, scaled_score))
else:
    final_data = []
    
# (6) 最終スコアが低い順にソート
final_data.sort(key=lambda x: x[1])

# (7) 結果をファイルに保存
with open(output_filename, "w", encoding="utf-8") as f:
    for ngram_string, score in final_data:
        f.write(f"{ngram_string}\t{score}\n")

print(f"✅ Done! Saved combined scores to '{output_filename}'")
