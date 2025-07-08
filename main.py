# (0) ライブラリのインポート
from collections import Counter
import numpy as np
from tqdm.auto import tqdm
import nltk
from nltk.corpus import stopwords
from datasets import load_dataset

# --------------------------------------------------------------------------
# ★★★ 設定項目 ★★★
N_GRAM_SIZE = 1
SCORE_TYPE = 'cost'
# --------------------------------------------------------------------------

# --- 設定チェックとファイル名設定 ---
if N_GRAM_SIZE != 1:
    print("Warning: This script is optimized for N_GRAM_SIZE = 1 to include POS tags.")
    print("For N-grams > 1, the output will only be the N-gram and its score.")
output_filename = f"{N_GRAM_SIZE}-grams_score_{SCORE_TYPE}_pos_combined.txt"

# --- NLTKデータ準備 ---
# averaged_perceptron_tagger (品詞推定) と stopwords を追加
required_nltk_packages = ['punkt', 'averaged_perceptron_tagger', 'stopwords']
for package in required_nltk_packages:
    try:
        # パッケージの種類に応じて検索パスを切り替え
        if package == 'punkt':
            nltk.data.find(f'tokenizers/{package}')
        elif package == 'averaged_perceptron_tagger':
            nltk.data.find(f'taggers/{package}')
        else:
            nltk.data.find(f'corpora/{package}')
    except LookupError:
        print(f"Downloading NLTK package: {package}")
        nltk.download(package, quiet=True)

# (1) ストップワードとカウンターの準備
print("Loading NLTK resources...")
stop_words = set(stopwords.words('english'))
unigram_counts = Counter()
word_details = {} # 元の単語の形と品詞を保存する辞書
print("✅ NLTK resources loaded.")

# (2) データセットの準備 (ストリーミング)
print("Loading dataset metadata...")
# ストリーミングではlen()が使えないため、おおよそのサイズを手動で設定
# wikitext-103-v1のtrainスプリットには約180万ドキュメントがある
total_docs = 1810350
dataset = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True, trust_remote_code=True)

# (3) ★★★ 単語、品詞をカウント ★★★
print(f"Starting to count {N_GRAM_SIZE}-grams with POS tags...")
for item in tqdm(dataset, total=total_docs, desc="Docs processed"):
    text = item["text"]
    if not text.strip():
        continue
    
    tokens = nltk.word_tokenize(text)
    # 品詞を推定
    tagged_tokens = nltk.pos_tag(tokens)

    for word, tag in tagged_tokens:
        # アルファベットのみで、ストップワードでない単語を対象
        if word.isalpha() and word.lower() not in stop_words:
            lower_word = word.lower()
            unigram_counts[lower_word] += 1
            
            # 固有名詞(NNP)を優先して単語情報を保存
            if lower_word not in word_details or tag == 'NNP':
                word_details[lower_word] = {'original': word, 'pos': tag}

print("✅ Frequency counting complete.")


# (4) スコアを計算 (浮動小数点)
print(f"Calculating floating point '{SCORE_TYPE}' scores...")
float_scores_data = []

total_unigrams = sum(unigram_counts.values())
for lower_word, count in tqdm(unigram_counts.items(), desc="Calculating costs"):
    if count > 0 and lower_word in word_details:
        # コストを計算
        cost_score = -np.log(count / total_unigrams)
        details = word_details[lower_word]
        float_scores_data.append({
            "lower": lower_word,
            "original": details['original'],
            "pos": details['pos'],
            "score": cost_score
        })

# (5) スコアを0-65535の範囲に正規化
print("Normalizing scores to short integer range (0-65535)...")
if float_scores_data:
    scores = [item['score'] for item in float_scores_data]
    min_score, max_score = min(scores), max(scores)
    score_range = max_score - min_score
    if score_range == 0: score_range = 1 # ゼロ除算を防止

    final_data = []
    for item in float_scores_data:
        scaled_score = int(((item['score'] - min_score) / score_range) * 65535)
        item['scaled_score'] = scaled_score
        final_data.append(item)
else:
    final_data = []
    
# (6) 最終スコアが低い順にソート
final_data.sort(key=lambda x: x['scaled_score'])

# (7) 結果をファイルに保存
print(f"Saving results to '{output_filename}'...")
with open(output_filename, "w", encoding="utf-8") as f:
    # ヘッダーを書き込み
    f.write("input_word\toutput_word\tpos_tag\tscore\n")
    for item in final_data:
        f.write(f"{item['lower']}\t{item['original']}\t{item['pos']}\t{item['scaled_score']}\n")

print(f"✅ Done! Saved combined scores with POS tags to '{output_filename}'")
