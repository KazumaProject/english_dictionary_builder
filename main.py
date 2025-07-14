# (0) ライブラリのインポート
from datasets import load_dataset
from collections import Counter
import numpy as np
from tqdm.auto import tqdm
import spacy

# --------------------------------------------------------------------------
# ★★★ 設定項目 ★★★
N_GRAM_SIZE = 1
SCORE_TYPE = 'cost'
DATASET_CONFIGS = [
    # より小さい wikitext データセットでテスト
    {"name": "wikitext", "config": "wikitext-103-v1", "split": "train", "column": "text"},
]
# --------------------------------------------------------------------------

output_filename = f"{N_GRAM_SIZE}-grams_score_{SCORE_TYPE}_pos_combined_with_ner.txt"

# (1) spaCyモデルの読み込み
# ★★★ 変更点: nerを有効化（リストから削除）★★★
# 固有表現抽出(NER)を有効にし、より高精度に固有名詞を抽出
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm", disable=["parser"])
print("✅ spaCy model loaded.")

# (2) カウンターとデータ保持用辞書を初期化
unigram_counts = Counter()
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

    # nlp.pipeを使用してテキストを効率的に一括処理
    # documentsからテキストを抽出するジェネレータを作成
    texts_generator = (item[config["column"]] for item in dataset)
    
    # バッチサイズを設定 (マシンのメモリに応じて調整)
    batch_size = 500

    # nlp.pipeで処理。tqdmで進捗を表示
    for doc in tqdm(nlp.pipe(texts_generator, batch_size=batch_size), desc=f"Processing {dataset_name}"):
        words_to_count = []
        
        # ★★★ 変更点: 固有表現(Entities)を先に処理 ★★★
        processed_entity_tokens = set()
        for ent in doc.ents:
            # GPE (地名), PERSON (人名), ORG (組織名) などの固有名詞を優先的に処理
            if ent.label_ in ['GPE', 'PERSON', 'ORG', 'PRODUCT', 'LOC', 'FAC', 'EVENT', 'WORK_OF_ART']:
                # 固有名詞が複数のトークンからなる場合、中心となるトークン(root)を代表として扱う
                root_token = ent.root
                if root_token.is_alpha and not root_token.is_stop:
                    lower_word = root_token.lower_
                    words_to_count.append(lower_word)
                    
                    # 固有表現ラベル(GPE, PERSON等)を品詞として保存
                    word_details[lower_word] = {'original': root_token.text, 'pos': ent.label_}
                    processed_entity_tokens.add(root_token)

        # 次に、通常のトークンを処理
        for token in doc:
            # 既に固有表現として処理されたトークンはスキップ
            if token in processed_entity_tokens:
                continue

            if token.is_alpha and not token.is_stop:
                lower_word = token.lower_
                words_to_count.append(lower_word)

                # まだ詳細が記録されていない単語のみ記録する
                # (固有表現が優先されるため、通常のPROPNで上書きされるのを防ぐ)
                if lower_word not in word_details:
                    word_details[lower_word] = {'original': token.text, 'pos': token.pos_}
        
        unigram_counts.update(words_to_count)

print("\n✅ All datasets processed. Frequency counting complete.")


# (4) スコアを計算 (浮動小数点)
print(f"Calculating floating point '{SCORE_TYPE}' scores...")
float_scores_data = []

total_unigrams = sum(unigram_counts.values())
for lower_word, count in tqdm(unigram_counts.items(), desc="Calculating costs"):
    if count > 0 and lower_word in word_details:
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

print(f"✅ Done! Saved combined scores with POS/NER tags to '{output_filename}'")
