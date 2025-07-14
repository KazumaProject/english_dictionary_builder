# main.py

from datasets import load_dataset
from collections import Counter
import numpy as np
from tqdm.auto import tqdm
import spacy

# --------------------------------------------------------------------------
# ★★★ 設定項目 ★★★
N_GRAM_SIZE = 2
SCORE_TYPE = 'cost'
DATASET_CONFIGS = [
    # より小さい wikitext データセットでテスト
    {"name": "wikitext", "config": "wikitext-103-v1", "split": "train", "column": "text"},
    # {"name": "wikipedia", "config": "20220301.en", "split": "train", "column": "text"},
]
# --------------------------------------------------------------------------

output_filename = f"{N_GRAM_SIZE}-grams_score_{SCORE_TYPE}_pos_combined_with_ner.txt"

# (1) spaCyモデルの読み込み
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm", disable=["parser"])
print("✅ spaCy model loaded.")

# (2) カウンターとデータ保持用辞書を初期化
ngram_counts = Counter()
word_details = {}

# (3) ★★★ 複数のデータセットを順番に処理 ★★★
for config in DATASET_CONFIGS:
    dataset_name = config["name"]
    print(f"\nProcessing dataset: {dataset_name} (split: {config['split']})...")

    dataset = load_dataset(
        dataset_name,
        config.get("config"),
        split=config["split"],
        streaming=True,
        trust_remote_code=True
    )

    texts_generator = (item[config["column"]] for item in dataset)
    batch_size = 500

    for doc in tqdm(nlp.pipe(texts_generator, batch_size=batch_size), desc=f"Processing {dataset_name}"):
        tokens = []
        processed_entity_tokens = set()

        # 固有表現(Entities)を先に処理
        for ent in doc.ents:
            if ent.label_ in ['GPE', 'PERSON', 'ORG', 'PRODUCT', 'LOC', 'FAC', 'EVENT', 'WORK_OF_ART']:
                root = ent.root
                if root.is_alpha and not root.is_stop:
                    lw = root.lower_
                    tokens.append(lw)
                    word_details[lw] = {'original': root.text, 'pos': ent.label_}
                    processed_entity_tokens.add(root)

        # 通常のトークンを処理
        for token in doc:
            if token in processed_entity_tokens:
                continue
            if token.is_alpha and not token.is_stop:
                lw = token.lower_
                tokens.append(lw)
                if lw not in word_details:
                    word_details[lw] = {'original': token.text, 'pos': token.pos_}

        # n-gramをカウント
        if N_GRAM_SIZE == 1:
            ngram_counts.update(tokens)
        else:
            if len(tokens) >= N_GRAM_SIZE:
                ngrams = [
                    " ".join(tokens[i:i+N_GRAM_SIZE])
                    for i in range(len(tokens) - N_GRAM_SIZE + 1)
                ]
                ngram_counts.update(ngrams)
                for gram in ngrams:
                    if gram not in word_details:
                        word_details[gram] = {'original': gram, 'pos': 'NGRAM'}

print("\n✅ All datasets processed. Frequency counting complete.")

# (4) スコアを計算 (浮動小数点)
print(f"Calculating floating point '{SCORE_TYPE}' scores...")
float_scores_data = []
total_ngrams = sum(ngram_counts.values())
for gram, cnt in tqdm(ngram_counts.items(), desc="Calculating costs"):
    if cnt > 0 and gram in word_details:
        cost_score = -np.log(cnt / total_ngrams)
        details = word_details[gram]
        float_scores_data.append({
            "lower": gram,
            "original": details['original'],
            "pos": details['pos'],
            "score": cost_score
        })

# (5) スコアを0-65535の範囲に正規化
print("Normalizing scores to short integer range (0-65535)...")
if float_scores_data:
    scores = [item['score'] for item in float_scores_data]
    min_score, max_score = min(scores), max(scores)
    score_range = max_score - min_score or 1

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

print(f"✅ Done! Saved combined scores with POS/NER tags to '{output_filename}'")
