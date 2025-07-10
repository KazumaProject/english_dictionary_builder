# (0) ライブラリのインポート
from datasets import load_dataset
from collections import Counter
import numpy as np
from tqdm.auto import tqdm
import spacy
from itertools import islice

# --------------------------------------------------------------------------
# ★★★ 設定項目 ★★★
N_GRAM_SIZE = 1
SCORE_TYPE = 'cost'
# 処理する最大サンプル数。GitHub Actionsの実行時間（最大6時間）に応じて調整してください。
# 100万サンプルあたり、おおよそ1〜2時間かかる可能性があります。
MAX_SAMPLES_TO_PROCESS = 1_000_000 

DATASET_CONFIGS = [
    # C4データセットを使用 (英語)
    {"name": "allenai/c4", "config": "en", "split": "train", "column": "text"},
]
# --------------------------------------------------------------------------

output_filename = f"{N_GRAM_SIZE}-grams_score_{SCORE_TYPE}_pos_combined_c4_subset.txt"

# (1) spaCyモデルの読み込み
# 高速化のために不要なパイプライン（parser, ner）を無効化
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
print("✅ spaCy model loaded.")

# (2) カウンターとデータ保持用辞書を初期化
unigram_counts = Counter()
word_details = {}

# (3) ★★★ 複数のデータセットを順番に処理 ★★★
for config in DATASET_CONFIGS:
    dataset_name = config["name"]
    print(f"\nProcessing dataset: {dataset_name} (config: {config['config']}, split: {config['split']})...")
    print(f"Processing up to {MAX_SAMPLES_TO_PROCESS:,} samples.")

    # ストリーミングモードでデータセットを読み込む
    dataset = load_dataset(
        path=dataset_name, 
        name=config.get("config"), 
        split=config["split"], 
        streaming=True, 
        trust_remote_code=True
    )

    # dataset.take() を使ってデータセットの先頭から指定した数だけ取り出す
    subset_dataset = dataset.take(MAX_SAMPLES_TO_PROCESS)

    # nlp.pipeを使用してテキストを効率的に一括処理
    # documentsからテキストを抽出するジェネレータを作成
    texts_generator = (item[config["column"]] for item in subset_dataset)
    
    # バッチサイズを設定 (マシンのメモリに応じて調整)
    batch_size = 500

    # tqdmに合計値を設定して進捗バーを正確に表示
    progress_bar = tqdm(
        nlp.pipe(texts_generator, batch_size=batch_size), 
        desc=f"Processing {dataset_name}",
        total=MAX_SAMPLES_TO_PROCESS
    )

    for doc in progress_bar:
        words_to_count = []
        for token in doc:
            # is_alphaで英字のみを対象とし、stopワードを除外
            if token.is_alpha and not token.is_stop:
                lower_word = token.lower_
                words_to_count.append(lower_word)

                # 固有名詞(PROPN)を優先して単語情報を保存
                if lower_word not in word_details or token.pos_ == 'PROPN':
                    word_details[lower_word] = {'original': token.text, 'pos': token.pos_}
        
        unigram_counts.update(words_to_count)

print("\n✅ All datasets processed. Frequency counting complete.")


# (4) スコアを計算 (浮動小数点)
print(f"Calculating floating point '{SCORE_TYPE}' scores...")
float_scores_data = []

total_unigrams = sum(unigram_counts.values())
if total_unigrams > 0:
    for lower_word, count in tqdm(unigram_counts.items(), desc="Calculating costs"):
        if count > 0 and lower_word in word_details:
            # ゼロ除算を避ける
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
    f.write("input_word\toutput_word\tpos_tag\tscore\n")
    for item in final_data:
        f.write(f"{item['lower']}\t{item['original']}\t{item['pos']}\t{item['scaled_score']}\n")

print(f"✅ Done! Saved combined scores with POS tags to '{output_filename}'")
