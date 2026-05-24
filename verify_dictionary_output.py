import json
import sys
import zipfile
from collections import Counter
from pathlib import Path


OUTPUT_FILENAME = "1-grams_score_cost_pos_combined_with_ner.txt"
ZIP_FILENAME = "1-grams_score_cost_pos_combined_with_ner.zip"
SUMMARY_FILENAME = "dictionary_source_summary.json"
EXPECTED_HEADER = ["input_word", "output_word", "pos_tag", "score"]
REQUIRED_PAIRS = [
    ("because", "because"),
    ("the", "the"),
    ("and", "and"),
    ("you", "you"),
    ("i", "I"),
    ("is", "is"),
    ("are", "are"),
    ("was", "was"),
    ("were", "were"),
    ("have", "have"),
    ("has", "has"),
    ("do", "do"),
    ("does", "does"),
    ("can", "can"),
    ("will", "will"),
    ("would", "would"),
    ("should", "should"),
    ("this", "this"),
    ("that", "that"),
    ("with", "with"),
    ("from", "from"),
    ("about", "about"),
    ("im", "I'm"),
    ("dont", "don't"),
    ("cant", "can't"),
    ("color", "color"),
    ("colour", "colour"),
    ("favorite", "favorite"),
    ("favourite", "favourite"),
    ("app", "app"),
    ("email", "email"),
    ("password", "password"),
    ("android", "Android"),
    ("github", "GitHub"),
    ("openai", "OpenAI"),
    ("autocad", "AutoCAD"),
]


def fail(message):
    print(f"FAIL: {message}")
    sys.exit(1)


def read_dictionary_text():
    txt_path = Path(OUTPUT_FILENAME)
    if txt_path.exists():
        return txt_path.read_text(encoding="utf-8"), str(txt_path)

    zip_path = Path(ZIP_FILENAME)
    if not zip_path.exists():
        fail(f"Neither {OUTPUT_FILENAME} nor {ZIP_FILENAME} exists")
    with zipfile.ZipFile(zip_path) as zf:
        if OUTPUT_FILENAME not in zf.namelist():
            fail(f"{ZIP_FILENAME} does not contain {OUTPUT_FILENAME}")
        return zf.read(OUTPUT_FILENAME).decode("utf-8"), f"{ZIP_FILENAME}:{OUTPUT_FILENAME}"


def load_summary():
    summary_path = Path(SUMMARY_FILENAME)
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8")), str(summary_path)

    zip_path = Path(ZIP_FILENAME)
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            if SUMMARY_FILENAME in zf.namelist():
                return json.loads(zf.read(SUMMARY_FILENAME).decode("utf-8")), f"{ZIP_FILENAME}:{SUMMARY_FILENAME}"
    return None, None


def verify_rows(text):
    lines = text.splitlines()
    if not lines:
        fail("dictionary output is empty")

    header = lines[0].split("\t")
    if header != EXPECTED_HEADER:
        fail(f"header must be {'/'.join(EXPECTED_HEADER)}, got {header}")

    keys = set()
    input_words = set()
    pair_counts = Counter()

    for line_number, line in enumerate(lines[1:], start=2):
        parts = line.split("\t")
        if len(parts) != 4:
            fail(f"line {line_number} must have 4 columns, got {len(parts)}")
        input_word, output_word, pos_tag, score_text = parts
        if not input_word:
            fail(f"line {line_number} has empty input_word")
        if not output_word:
            fail(f"line {line_number} has empty output_word")
        if not pos_tag:
            fail(f"line {line_number} has empty pos_tag")
        try:
            score = int(score_text)
        except ValueError:
            fail(f"line {line_number} score is not an integer: {score_text!r}")
        if not 0 <= score <= 32767:
            fail(f"line {line_number} score is out of range: {score}")
        key = (input_word, output_word)
        if key in keys:
            fail(f"duplicate key found: {input_word}\\t{output_word}")
        keys.add(key)
        input_words.add(input_word)
        pair_counts[key] += 1

    if len(input_words) < 50000:
        fail(f"unique input_word count is below 50000: {len(input_words)}")

    missing = [pair for pair in REQUIRED_PAIRS if pair not in keys]
    if missing:
        fail("required dictionary pairs missing: " + ", ".join(f"{a}\\t{b}" for a, b in missing))

    if len(input_words) >= 200000:
        coverage = "high coverage"
    elif len(input_words) >= 100000:
        coverage = "standard coverage"
    else:
        coverage = "minimum coverage"

    return {
        "rows": len(lines) - 1,
        "unique_input_words": len(input_words),
        "coverage": coverage,
        "required_pairs_checked": len(REQUIRED_PAIRS),
    }


def verify_summary(summary):
    if summary is None:
        fail(f"{SUMMARY_FILENAME} is missing")
    sources = summary.get("sources")
    if not isinstance(sources, dict):
        fail("dictionary_source_summary.json must contain a sources object")

    wordfreq = sources.get("wordfreq")
    if not isinstance(wordfreq, dict):
        fail("dictionary_source_summary.json must contain sources.wordfreq")
    valid_words = int(wordfreq.get("valid_words", 0))
    if valid_words < 50000:
        fail(f"wordfreq layer valid_words is below 50000: {valid_words}")

    requested_top_n = int(wordfreq.get("requested_top_n", 0))
    if requested_top_n == 100000 and valid_words < 90000:
        fail(f"WORD_FREQ_TOP_N=100000 but wordfreq valid_words is unexpectedly low: {valid_words}")

    merge = summary.get("merge")
    if not isinstance(merge, dict):
        fail("dictionary_source_summary.json must contain a merge object")
    if "source_counts_before_merge" not in merge or "source_counts_after_merge" not in merge:
        fail("dictionary_source_summary.json must include source counts before and after merge")

    return {
        "wordfreq_valid_words": valid_words,
        "wordfreq_requested_top_n": requested_top_n,
        "summary_unique_input_words": int(merge.get("unique_input_words", 0)),
        "source_counts_before_merge": merge["source_counts_before_merge"],
        "source_counts_after_merge": merge["source_counts_after_merge"],
    }


def main():
    text, dictionary_source = read_dictionary_text()
    row_result = verify_rows(text)
    summary, summary_source = load_summary()
    summary_result = verify_summary(summary)

    print(f"Dictionary source: {dictionary_source}")
    print(f"Summary source: {summary_source}")
    print(f"Rows: {row_result['rows']}")
    print(f"Unique input words: {row_result['unique_input_words']} ({row_result['coverage']})")
    print(f"Required pairs checked: {row_result['required_pairs_checked']}")
    print(f"wordfreq valid words: {summary_result['wordfreq_valid_words']}")
    print("Source counts before merge:", summary_result["source_counts_before_merge"])
    print("Source counts after merge:", summary_result["source_counts_after_merge"])
    print("OK: dictionary output verification passed")


if __name__ == "__main__":
    main()
