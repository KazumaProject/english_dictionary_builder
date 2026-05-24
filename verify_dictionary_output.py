import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path


OUTPUT_FILENAME = "1-grams_score_cost_pos_combined_with_ner.txt"
ZIP_FILENAME = "1-grams_score_cost_pos_combined_with_ner.zip"
PHRASE_OUTPUT_FILENAME = "english_phrases.tsv"
PHRASE_ZIP_FILENAME = "english_phrases.zip"
SUMMARY_FILENAME = "dictionary_source_summary.json"
EXPECTED_UNIGRAM_HEADER = ["input_word", "output_word", "pos_tag", "score"]
EXPECTED_PHRASE_HEADER = ["input_phrase", "output_phrase", "pos_tag", "score"]
LOWER_ASCII_ALPHA_RE = re.compile(r"^[a-z]+$")

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

REQUIRED_PHRASE_PAIRS = [
    ("thankyou", "thank you"),
    ("letmeknow", "let me know"),
    ("howareyou", "how are you"),
    ("goodmorning", "good morning"),
    ("goodnight", "good night"),
    ("androidkeyboard", "Android keyboard"),
    ("japanesekeyboard", "Japanese keyboard"),
    ("inputmethod", "input method"),
    ("inputmethodeditor", "Input Method Editor"),
    ("opensource", "open source"),
    ("githubactions", "GitHub Actions"),
    ("pullrequest", "pull request"),
]


def fail(message):
    print(f"FAIL: {message}")
    sys.exit(1)


def read_text_from_file_or_zip(filename, zip_filename):
    txt_path = Path(filename)
    if txt_path.exists():
        return txt_path.read_text(encoding="utf-8"), str(txt_path)

    zip_path = Path(zip_filename)
    if not zip_path.exists():
        fail(f"Neither {filename} nor {zip_filename} exists")
    with zipfile.ZipFile(zip_path) as zf:
        if filename not in zf.namelist():
            fail(f"{zip_filename} does not contain {filename}")
        return zf.read(filename).decode("utf-8"), f"{zip_filename}:{filename}"


def load_summary():
    summary_path = Path(SUMMARY_FILENAME)
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8")), str(summary_path)

    for zip_filename in [ZIP_FILENAME, PHRASE_ZIP_FILENAME]:
        zip_path = Path(zip_filename)
        if zip_path.exists():
            with zipfile.ZipFile(zip_path) as zf:
                if SUMMARY_FILENAME in zf.namelist():
                    return json.loads(zf.read(SUMMARY_FILENAME).decode("utf-8")), f"{zip_filename}:{SUMMARY_FILENAME}"
    return None, None


def verify_zip_contents():
    required_zip_contents = {
        ZIP_FILENAME: [OUTPUT_FILENAME, SUMMARY_FILENAME],
        PHRASE_ZIP_FILENAME: [PHRASE_OUTPUT_FILENAME, SUMMARY_FILENAME],
    }

    for zip_filename, required_names in required_zip_contents.items():
        zip_path = Path(zip_filename)
        if not zip_path.exists():
            fail(f"{zip_filename} is missing")
        with zipfile.ZipFile(zip_path) as zf:
            names = set(zf.namelist())
            missing = [name for name in required_names if name not in names]
            if missing:
                fail(f"{zip_filename} is missing required file(s): {', '.join(missing)}")


def parse_score(score_text, line_number, label):
    try:
        score = int(score_text)
    except ValueError:
        fail(f"{label} line {line_number} score is not an integer: {score_text!r}")
    if not 0 <= score <= 32767:
        fail(f"{label} line {line_number} score is out of range: {score}")
    return score


def verify_unigram_rows(text):
    lines = text.splitlines()
    if not lines:
        fail("unigram dictionary output is empty")

    header = lines[0].split("\t")
    if header != EXPECTED_UNIGRAM_HEADER:
        fail(f"unigram header must be {'/'.join(EXPECTED_UNIGRAM_HEADER)}, got {header}")

    keys = set()
    input_words = set()
    pair_counts = Counter()

    for line_number, line in enumerate(lines[1:], start=2):
        parts = line.split("\t")
        if len(parts) != 4:
            fail(f"unigram line {line_number} must have 4 columns, got {len(parts)}")
        input_word, output_word, pos_tag, score_text = parts
        if not input_word:
            fail(f"unigram line {line_number} has empty input_word")
        if not output_word:
            fail(f"unigram line {line_number} has empty output_word")
        if not pos_tag:
            fail(f"unigram line {line_number} has empty pos_tag")
        parse_score(score_text, line_number, "unigram")
        key = (input_word, output_word)
        if key in keys:
            fail(f"duplicate unigram key found: {input_word}\\t{output_word}")
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


def verify_phrase_rows(text):
    lines = text.splitlines()
    if not lines:
        fail("phrase dictionary output is empty")

    header = lines[0].split("\t")
    if header != EXPECTED_PHRASE_HEADER:
        fail(f"phrase header must be {'/'.join(EXPECTED_PHRASE_HEADER)}, got {header}")

    keys = set()
    input_phrases = set()

    for line_number, line in enumerate(lines[1:], start=2):
        parts = line.split("\t")
        if len(parts) != 4:
            fail(f"phrase line {line_number} must have 4 columns, got {len(parts)}")
        input_phrase, output_phrase, pos_tag, score_text = parts
        if not input_phrase:
            fail(f"phrase line {line_number} has empty input_phrase")
        if not output_phrase:
            fail(f"phrase line {line_number} has empty output_phrase")
        if not pos_tag:
            fail(f"phrase line {line_number} has empty pos_tag")
        if not LOWER_ASCII_ALPHA_RE.fullmatch(input_phrase):
            fail(f"phrase line {line_number} input_phrase must be lowercase ascii alpha only: {input_phrase!r}")
        if "\t" in output_phrase or "\n" in output_phrase or "\r" in output_phrase:
            fail(f"phrase line {line_number} output_phrase contains a forbidden control character")
        parse_score(score_text, line_number, "phrase")
        key = (input_phrase, output_phrase)
        if key in keys:
            fail(f"duplicate phrase key found: {input_phrase}\\t{output_phrase}")
        keys.add(key)
        input_phrases.add(input_phrase)

    missing = [pair for pair in REQUIRED_PHRASE_PAIRS if pair not in keys]
    if missing:
        fail("required phrase pairs missing: " + ", ".join(f"{a}\\t{b}" for a, b in missing))

    return {
        "rows": len(lines) - 1,
        "unique_input_phrases": len(input_phrases),
        "required_phrase_pairs_checked": len(REQUIRED_PHRASE_PAIRS),
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
    unigram_merge = merge.get("unigram")
    phrase_merge = merge.get("phrase")
    if not isinstance(unigram_merge, dict):
        fail("dictionary_source_summary.json must contain merge.unigram")
    if not isinstance(phrase_merge, dict):
        fail("dictionary_source_summary.json must contain merge.phrase")
    for label, merge_object in [("unigram", unigram_merge), ("phrase", phrase_merge)]:
        if "source_counts_before_merge" not in merge_object or "source_counts_after_merge" not in merge_object:
            fail(f"dictionary_source_summary.json must include {label} source counts before and after merge")

    for key in ["manual_inflection_lemmas", "manual_inflection_overrides"]:
        if not isinstance(sources.get(key), dict):
            fail(f"dictionary_source_summary.json must contain sources.{key}")

    inflections = sources.get("inflections")
    if not isinstance(inflections, dict):
        fail("dictionary_source_summary.json must contain sources.inflections")
    accepted_count = inflections.get("accepted_count")
    if not isinstance(accepted_count, int):
        fail("sources.inflections.accepted_count must be present")
    validation_sources = inflections.get("validation_sources")
    if not isinstance(validation_sources, list) or not validation_sources:
        fail("sources.inflections.validation_sources must be present")
    accepted_source_counts = inflections.get("accepted_source_counts")
    if not isinstance(accepted_source_counts, dict):
        fail("sources.inflections.accepted_source_counts must be present")
    accepted_entry_source_counts = inflections.get("accepted_entry_source_counts")
    if not isinstance(accepted_entry_source_counts, dict) or "override" not in accepted_entry_source_counts:
        fail("sources.inflections must record irregular override-derived entries")
    for sample in inflections.get("accepted_samples", []):
        sample_sources = sample.get("validation_sources")
        if not isinstance(sample_sources, list) or not sample_sources:
            fail("accepted inflection samples must include validation_sources")

    if not isinstance(sources.get("manual_phrases"), dict):
        fail("dictionary_source_summary.json must contain sources.manual_phrases")
    if not isinstance(sources.get("corpus_bigram"), dict):
        fail("dictionary_source_summary.json must contain sources.corpus_bigram")

    return {
        "wordfreq_valid_words": valid_words,
        "wordfreq_requested_top_n": requested_top_n,
        "unigram_unique_input_words": int(unigram_merge.get("unique_input_words", 0)),
        "phrase_unique_input_phrases": int(phrase_merge.get("unique_input_phrases", 0)),
        "unigram_source_counts_before_merge": unigram_merge["source_counts_before_merge"],
        "unigram_source_counts_after_merge": unigram_merge["source_counts_after_merge"],
        "phrase_source_counts_before_merge": phrase_merge["source_counts_before_merge"],
        "phrase_source_counts_after_merge": phrase_merge["source_counts_after_merge"],
        "inflection_generated_candidate_count": int(inflections.get("generated_candidate_count", 0)),
        "inflection_accepted_count": accepted_count,
        "inflection_rejected_count": int(inflections.get("rejected_count", 0)),
    }


def main():
    unigram_text, dictionary_source = read_text_from_file_or_zip(OUTPUT_FILENAME, ZIP_FILENAME)
    phrase_text, phrase_source = read_text_from_file_or_zip(PHRASE_OUTPUT_FILENAME, PHRASE_ZIP_FILENAME)
    verify_zip_contents()

    unigram_result = verify_unigram_rows(unigram_text)
    phrase_result = verify_phrase_rows(phrase_text)
    summary, summary_source = load_summary()
    summary_result = verify_summary(summary)

    print(f"Dictionary source: {dictionary_source}")
    print(f"Phrase source: {phrase_source}")
    print(f"Summary source: {summary_source}")
    print(f"Rows: {unigram_result['rows']}")
    print(f"Unique input words: {unigram_result['unique_input_words']} ({unigram_result['coverage']})")
    print(f"Required pairs checked: {unigram_result['required_pairs_checked']}")
    print(f"Phrase rows: {phrase_result['rows']}")
    print(f"Unique input phrases: {phrase_result['unique_input_phrases']}")
    print(f"Required phrase pairs checked: {phrase_result['required_phrase_pairs_checked']}")
    print(f"wordfreq valid words: {summary_result['wordfreq_valid_words']}")
    print(
        "Inflections:",
        {
            "generated_candidate_count": summary_result["inflection_generated_candidate_count"],
            "accepted_count": summary_result["inflection_accepted_count"],
            "rejected_count": summary_result["inflection_rejected_count"],
        },
    )
    print("Unigram source counts before merge:", summary_result["unigram_source_counts_before_merge"])
    print("Unigram source counts after merge:", summary_result["unigram_source_counts_after_merge"])
    print("Phrase source counts before merge:", summary_result["phrase_source_counts_before_merge"])
    print("Phrase source counts after merge:", summary_result["phrase_source_counts_after_merge"])
    print("OK: dictionary output verification passed")


if __name__ == "__main__":
    main()
