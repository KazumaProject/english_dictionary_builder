# (0) ライブラリのインポート
from datasets import load_dataset
from collections import Counter
import numpy as np
from tqdm.auto import tqdm
import spacy
import itertools

# --------------------------------------------------------------------------
# ★★★ 設定項目 ★★★
N_GRAM_SIZE = 1
SCORE_TYPE = 'cost'
DATASET_CONFIGS = [
    {"name": "wikitext", "config": "wikitext-103-v1", "split": "train", "column": "text"},
    {"name": "wikipedia", "config": "20220301.en", "split": "train", "column": "text"},
]
# [改善] 最小出現回数のしきい値を設定
MIN_FREQUENCY = 5

# [改善] GitHub Actionsの無料枠で実行するための最大アイテム数
# この値を調整して実行時間を6時間以内に収める
# まずは10万~50万件で試し、実行時間を見て調整してください
MAX_ITEMS_PER_DATASET = 500_000
# --------------------------------------------------------------------------

output_filename = f"{N_GRAM_SIZE}-grams_score_{SCORE_TYPE}_pos_combined.txt"

# (1) spaCyモデルの読み込み
print("Loading spaCy model...")
# [改善] より高精度なモデルを使用
nlp = spacy.load("en_core_web_lg", disable=["parser", "ner"])
print("✅ spaCy model loaded.")

# (2) カウンターとデータ保持用辞書を初期化
unigram_counts = Counter()
word_details = {}

# (3) ★★★ 複数のデータセットを順番に処理 ★★★
for i, config in enumerate(DATASET_CONFIGS):
    dataset_name = config["name"]
    print(f"\nProcessing dataset: {dataset_name} (split: {config['split']})...")

    dataset = load_dataset(
        dataset_name,
        config.get("config"),
        split=config["split"],
        streaming=True,
        trust_remote_code=True
    )

    # [改善] isliceを使ってデータ数を制限
    texts_generator = (item[config["column"]] for item in itertools.islice(dataset, MAX_ITEMS_PER_DATASET))

    # バッチサイズを設定
    batch_size = 500

    # nlp.pipeで処理。tqdmで進捗を表示
    for doc in tqdm(nlp.pipe(texts_generator, batch_size=batch_size), total=MAX_ITEMS_PER_DATASET, desc=f"Processing {dataset_name}"):
        words_to_count = []
        for token in doc:
            # is_alphaで英字のみを対象とし、stopワードを除外
            if token.is_alpha and not token.is_stop:
                # [改善] .lower_ の代わりに .lemma_ を使用して見出し語化
                lemma = token.lemma_
                words_to_count.append(lemma)

                # 固有名詞(PROPN)を優先して単語情報を保存
                if lemma not in word_details or token.pos_ == 'PROPN':
                    word_details[lemma] = {'original': token.text, 'pos': token.pos_}

        unigram_counts.update(words_to_count)

    # ★★★ [NEW] MEMORY OPTIMIZATION ★★★
    # Prune low-frequency words after processing each dataset (except the last one)
    # to free up memory before the next, potentially larger dataset.
    if i < len(DATASET_CONFIGS) - 1:
        print(f"\nOptimizing memory: Pruning low-frequency words from '{dataset_name}'...")
        # Identify lemmas with a count of 1, as they are least likely to reach MIN_FREQUENCY
        lemmas_to_prune = [lemma for lemma, count in unigram_counts.items() if count == 1]

        for lemma in tqdm(lemmas_to_prune, desc="Pruning single-occurrence words"):
            if unigram_counts.get(lemma) == 1:
                del unigram_counts[lemma]
                if lemma in word_details:
                    del word_details[lemma]
        print(f"✅ Pruned {len(lemmas_to_prune):,} words to save memory.")


print("\n✅ All datasets processed. Frequency counting complete.")


# (4) スコアを計算 (浮動小数点)
print(f"Calculating floating point '{SCORE_TYPE}' scores...")
float_scores_data = []

total_unigrams = sum(unigram_counts.values())
for lemma, count in tqdm(unigram_counts.items(), desc="Calculating scores"):
    # [改善] 最小出現回数でフィルタリング
    if count > MIN_FREQUENCY and lemma in word_details:
        cost_score = -np.log(count / total_unigrams)
        details = word_details[lemma]
        float_scores_data.append({
            "lower": lemma,
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
