# N‑gram Score Calculator / N‑グラムスコア計算ツール

> **English follows Japanese**

---

## 概要 (Overview – JP)

本プロジェクトは、Wikipedia などの大規模コーパスからユニグラム（1‑gram）の出現頻度を解析し、確率に基づくコストスコアを付与した TXT ファイルを生成する Python スクリプトと CI/CD ワークフローを提供します。spaCy の固有表現抽出 (NER) を活用することで、固有名詞を高精度に識別し、入力メソッド (IME) やスペルチェッカー、その他 NLP タスクに適した単語リストを作成できます。

## Overview (EN)

This project contains a high‑performance Python script that streams large text datasets (e.g. Wikipedia) from the Hugging Face Hub, counts unigram frequencies, applies a cost score `‑log(p)`, and normalises the result to the 0‑65535 range. Proper nouns are accurately captured via spaCy’s Named‑Entity Recognition (NER). The artefact is a ready‑to‑use TSV, suitable for IMEs, spell‑checkers, and other NLP pipelines.

---

## 特徴 (Features)

* **ストリーミング処理 / Streaming**: `datasets` ライブラリのストリーミング API を用いて、メモリ消費を最小化。
* **spaCy バッチ処理 / batched spaCy**: `nlp.pipe()` による高速バッチ解析。
* **固有表現優先 / Entity‑first**: GPE・PERSON など 8 種類の固有表現ラベルを優先してカウント。
* **確率コスト / Cost score**: `‑log(freq / total)` を 16‑bit 整数にスケール。
* **GitHub Actions**: タグ Push でスクリプトを実行し artefact (ZIP) を自動リリース。

---

## クイックスタート (Quick Start)

```bash
# 1. Clone
$ git clone <repo‑url>
$ cd n‑gram‑score‑calculator

# 2. Install (Python 3.10+)
$ pip install -r requirements.txt
$ python -m spacy download en_core_web_sm

# 3. Run
$ python main.py
# → 1-grams_score_cost_pos_combined_with_ner.txt が生成されます。
```

### 出力例 / Output Sample

```
input_word	output_word	pos_tag	score
apple	Apple	PROPN	32010
```

| Column (EN)   | 説明 (JP)            | Example |
| ------------- | ------------------ | ------- |
| `input_word`  | 小文字化した入力語          | `apple` |
| `output_word` | 元の表記（固有名詞は大文字）     | `Apple` |
| `pos_tag`     | spaCy 品詞 / POS tag | `PROPN` |
| `score`       | 正規化コスト (0‑65535)   | `32010` |

---

## 設定ファイル (Configuration)

```python
N_GRAM_SIZE = 1          # n in n‑gram (currently 1)
SCORE_TYPE  = 'cost'     # scoring method
DATASET_CONFIGS = [
    {"name":"wikitext","config":"wikitext-103-v1","split":"train","column":"text"},
]
```

`DATASET_CONFIGS` に辞書を追加するだけで別コーパスを簡単に追加できます。

---

## データソースとライセンス (Data Sources & Licenses)

このプロジェクトは外部の公開データセットをストリーミング取得して処理します。各データセットにはそれぞれ固有のライセンスが存在するため、使用・再配布時には必ず確認してください。

| Dataset             | License                                   | 注意事項                           |
| ------------------- | ----------------------------------------- | ------------------------------ |
| **wikitext‑103‑v1** | Derived from Wikipedia → **CC BY‑SA 3.0** | 派生物を配布する場合は同ライセンスでの共有と帰属表示が必要。 |

> **ワンポイント**: 本スクリプトが出力する成果物 は「個々の単語」とその頻度のみを含むため、一般的には著作物性が極めて低く、CC BY‑SA のコピーライト対象外と考えられます。ただし、大量の元本文を再配布する場合や、別データセットを追加する場合は、そのライセンス条項に従ってください。

---

## GitHub Actions

`/.github/workflows/release.yml` は以下を自動化します。

1. タグ Push (`v*.*.*`) をトリガー。
2. Python 環境をセットアップし、依存関係をインストール。
3. `main.py` を実行し TSV を生成。
4. ZIP 化して GitHub Release にアップロード。

---

## ライセンス (License)

Apache‑2.0.  詳細は `LICENSE` ファイルをご覧ください。
