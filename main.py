# (0) ライブラリのインポート
from datasets import load_dataset
from collections import Counter
import numpy as np
from tqdm.auto import tqdm
import nltk

# --------------------------------------------------------------------------
# ★★★ 設定項目 ★★★
# 1. 解析したいN-gramのサイズを指定 (1: unigram)
# ※このスクリプトは現在、unigram (N_GRAM_SIZE = 1) に最適化されています。
N_GRAM_SIZE = 1

# 2. スコアの種類を選択 ('cost' または 'pmi')
SCORE_TYPE = 'cost'

# 3. ★★★ 使用するデータセットのリスト ★★★
DATASET_CONFIGS = [
    {"name": "wikipedia", "config": "20220301.en", "split": "train", "column": "text"},
    # {"name": "bookcorpus", "config": None, "split": "train", "column": "text"},
    # {"name": "c4", "config": "en", "split": "train[:200000]", "column": "text"},
]
# --------------------------------------------------------------------------

# --- スクリプト設定チェック ---
if N_GRAM_SIZE > 1:
    print("⚠️  Warning: This script is currently optimized for unigrams (N_GRAM_SIZE = 1). The logic for POS tagging and preserving capitalization is not implemented for bigrams or trigrams.")

# --- ファイル名設定 ---
output_filename = f"{N_GRAM_SIZE}-grams_score_{SCORE_TYPE}_pos_combined.txt"

# --- NLTKデータ準備 ---
### 変更点 ###: POSタギング用の'averaged_perceptron_tagger'を追加
required_nltk_packages = ['punkt', 'words', 'averaged_perceptron_tagger']
for package in required_nltk_packages:
    try:
        if package == 'punkt':
            nltk.data.find(f'tokenizers/{package}')
        elif package == 'averaged_perceptron_tagger':
            nltk.data.find(f'taggers/{package}.zip')
        else:
            nltk.data.find(f'corpora/{package}.zip')
    except LookupError:
        nltk.download(package)

# (1) 英単語辞書をセットとして読み込み（今回は使用しませんが、参考のために残します）
# print("Loading English dictionary...")
# english_words = set(nltk.words())
# print(f"✅ Loaded {len(english_words)} words from English dictionary.")


# (2) カウンターとデータ保持用辞書を初期化
unigram_counts = Counter()
### 変更点 ###: 単語の元情報（大文字・小文字、品詞）を保持する辞書
word_details = {}

# (3) ★★★ 複数のデータセットを順番に処理 ★★★
for config in DATASET_CONFIGS:
    dataset_name = config["name"]
    print(f"\nProcessing dataset: {dataset_name} (split: {config['split']})...")

    total_docs = None # ストリーミングでは合計が取れないため

    dataset = load_dataset(
        dataset_name,
        config.get("config"),
        split=config["split"],
        streaming=True,
        trust_remote_code=True
    )

    for item in tqdm(dataset, total=total_docs, desc=f"Processing {dataset_name}"):
        ### 変更点 ###: テキストを小文字にせず、POSタギングを実行
        text = item[config["column"]]
        # word_tokenizeは文単位での処理が推奨されるため、文に分割
        sentences = nltk.sent_tokenize(text)
        
        for sentence in sentences:
            tokens = nltk.word_tokenize(sentence)
            tagged_tokens = nltk.pos_tag(tokens)

            words_to_count = []
            for word, tag in tagged_tokens:
                if word.isalpha():
                    lower_word = word.lower()
                    words_to_count.append(lower_word)

                    # 固有名詞(NNP, NNPS)を優先して単語情報を保存
                    # すでに登録されていても、固有名詞タグなら上書きする
                    if lower_word not in word_details or tag.startswith('NNP'):
                        word_details[lower_word] = {'original': word, 'pos': tag}

            if N_GRAM_SIZE == 1:
                unigram_counts.update(words_to_count)
            else:
                # N-gram > 1 の場合の処理 (今回は対象外)
                pass

print("\n✅ All datasets processed. Frequency counting complete.")


# (4) スコアを計算 (浮動小数点)
print(f"Calculating floating point '{SCORE_TYPE}' scores...")
float_scores_data = []

# 'cost'スコアの計算
total_unigrams = sum(unigram_counts.values())
for lower_word, count in tqdm(unigram_counts.items(), desc="Calculating costs"):
    if count > 0 and lower_word in word_details:
        # スコア計算
        cost_score = -np.log(count / total_unigrams)
        
        # 保持しておいた単語情報とスコアを結合
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
    if score_range == 0: score_range = 1

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
    # ヘッダーを書き込む
    f.write("input_word\toutput_word\tpos_tag\tscore\n")
    for item in final_data:
        f.write(f"{item['lower']}\t{item['original']}\t{item['pos']}\t{item['scaled_score']}\n")

print(f"✅ Done! Saved combined scores with POS tags to '{output_filename}'")
