# N-gram Score Calculator

## 概要 (Overview)

This project provides a script to generate a scored unigram list from large-scale text corpora like Wikipedia.

It's designed for high-performance processing by using the **spaCy** NLP library. The script identifies parts of speech (nouns, verbs, proper nouns, etc.) and preserves capitalization for proper nouns. The resulting list is ideal for input methods, spell checkers, and other natural language processing tasks.

## このスクリプトがしていること (What This Script Does)

The script performs the following steps:

1.  **Load Datasets**: It efficiently reads large text datasets from the Hugging Face Hub in streaming mode.

2.  **Process Text with spaCy**: It uses the `spaCy` library for fast and accurate text processing. The script processes text in large batches for maximum speed. For each word (token), it identifies:
    * The word itself.
    * Its Part-of-Speech (POS) tag (e.g., `PROPN` for proper noun, `VERB` for verb).
    * Whether it's a stop word (e.g., "the", "a", "is").

3.  **Count Frequencies & Store Data**:
    * It counts the frequency of each word in lowercase (excluding stop words).
    * It stores the canonical form of each word, prioritizing the capitalized version if it's identified as a proper noun (`PROPN`).

4.  **Calculate & Normalize Scores**:
    * A "cost" score (`-log(probability)`) is calculated for each word. More frequent words get a lower score.
    * Scores are normalized to an integer range of 0-65535.

5.  **Save Output**: The final list is sorted by score (most frequent first) and saved to a tab-separated text file (`.txt`).

---

## 成果物 (Artifact)

The script generates a `.txt` file with the following format.

**Filename**: `1-grams_score_cost_pos_combined.txt`

**Format**: Tab-Separated Values (TSV)

| カラム          | 説明                                                         | 例          |
| --------------- | ------------------------------------------------------------ | ----------- |
| `input_word`    | The lowercase input word.                                    | `apple`     |
| `output_word`   | The original cased word (e.g., proper nouns are capitalized).| `Apple`     |
| `pos_tag`       | The spaCy Part-of-Speech tag.                                | `PROPN`     |
| `score`         | The normalized cost score (0-65535); lower is more frequent. | `32010`     |

**Example:**
