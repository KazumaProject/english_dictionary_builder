import json
import math
import os
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from tqdm.auto import tqdm
from wordfreq import top_n_list, zipf_frequency


N_GRAM_SIZE = 1
SCORE_TYPE = "cost"
OUTPUT_FILENAME = f"{N_GRAM_SIZE}-grams_score_{SCORE_TYPE}_pos_combined_with_ner.txt"
ZIP_FILENAME = f"{N_GRAM_SIZE}-grams_score_{SCORE_TYPE}_pos_combined_with_ner.zip"
SUMMARY_FILENAME = "dictionary_source_summary.json"

WORD_FREQ_TOP_N = int(os.getenv("WORD_FREQ_TOP_N", "100000"))
WORD_FREQ_WORDLIST = os.getenv("WORD_FREQ_WORDLIST", "large")
ENABLE_CORPUS = os.getenv("ENABLE_CORPUS", "0").lower() in {"1", "true", "yes", "on"}
CORPUS_MAX_DOCS = int(os.getenv("CORPUS_MAX_DOCS", "0"))

DATASET_CONFIGS = [
    {"name": "wikitext", "config": "wikitext-103-v1", "split": "train", "column": "text"},
]

MANUAL_TSV_PATHS = [
    Path("data/manual/basic_words.tsv"),
    Path("data/manual/contractions.tsv"),
    Path("data/manual/spelling_variants.tsv"),
    Path("data/manual/tech_common_words.tsv"),
]

DENY_WORDS_PATH = Path("data/manual/deny_words.tsv")
SCOWL_WORDS_PATH = Path("external_sources/scowl/words.txt")

WORD_RE = re.compile(r"^[a-z]+$")
ENTITY_LABELS = {"GPE", "PERSON", "ORG", "PRODUCT", "LOC", "FAC", "EVENT", "WORK_OF_ART"}


@dataclass(frozen=True)
class Entry:
    input_word: str
    output_word: str
    pos_tag: str
    score: int
    source: str


def clamp_score(score):
    return max(0, min(32767, int(round(score))))


def is_allowed_input_word(word):
    if not word or not WORD_RE.fullmatch(word):
        return False
    if not 1 <= len(word) <= 32:
        return False
    if len(word) == 1 and word not in {"a", "i"}:
        return False
    return True


def normalize_wordfreq_word(word):
    if not word:
        return None
    normalized = word.strip().lower()
    if "'" in normalized or "\u2019" in normalized:
        return None
    if "<" in normalized or ">" in normalized or "http" in normalized:
        return None
    if not is_allowed_input_word(normalized):
        return None
    return normalized


def normalize_scowl_word(word):
    if not word:
        return None
    normalized = word.strip().lower()
    if "'" in normalized or "\u2019" in normalized:
        return None
    if not is_allowed_input_word(normalized):
        return None
    return normalized


def validate_entry(entry):
    if not entry.input_word or "\t" in entry.input_word or "\n" in entry.input_word:
        return False
    if not entry.output_word or "\t" in entry.output_word or "\n" in entry.output_word:
        return False
    if not entry.pos_tag or "\t" in entry.pos_tag or "\n" in entry.pos_tag:
        return False
    return 0 <= int(entry.score) <= 32767


def load_deny_words():
    words = set()
    read_lines = 0
    if not DENY_WORDS_PATH.exists():
        print(f"WARNING: deny words file not found: {DENY_WORDS_PATH}")
        return words, {"path": str(DENY_WORDS_PATH), "read_lines": 0, "valid_words": 0}

    with DENY_WORDS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            read_lines += 1
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            normalized = normalize_wordfreq_word(stripped)
            if normalized:
                words.add(normalized)

    print(f"Loaded deny words: read_lines={read_lines}, valid_words={len(words)}")
    return words, {"path": str(DENY_WORDS_PATH), "read_lines": read_lines, "valid_words": len(words)}


def load_manual_tsv(path):
    entries = []
    read_rows = 0
    excluded_rows = 0

    with Path(path).open("r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        if header != ["input_word", "output_word", "pos_tag", "score"]:
            raise ValueError(f"{path} must have header: input_word\\toutput_word\\tpos_tag\\tscore")

        for line_number, line in enumerate(f, start=2):
            read_rows += 1
            stripped = line.rstrip("\n")
            if not stripped:
                excluded_rows += 1
                continue
            parts = stripped.split("\t")
            if len(parts) != 4:
                raise ValueError(f"{path}:{line_number} must have 4 tab-separated columns")
            input_word, output_word, pos_tag, score_text = parts
            normalized_input = input_word.strip().lower()
            if not is_allowed_input_word(normalized_input):
                raise ValueError(f"{path}:{line_number} has invalid input_word: {input_word!r}")
            entry = Entry(
                input_word=normalized_input,
                output_word=output_word.strip(),
                pos_tag=pos_tag.strip(),
                score=clamp_score(int(score_text)),
                source=f"manual:{Path(path).stem}",
            )
            if not validate_entry(entry):
                raise ValueError(f"{path}:{line_number} has an invalid dictionary entry")
            entries.append(entry)

    print(f"Loaded manual TSV {path}: read_rows={read_rows}, valid_rows={len(entries)}, excluded_rows={excluded_rows}")
    return entries, {
        "path": str(path),
        "read_rows": read_rows,
        "valid_rows": len(entries),
        "excluded_rows": excluded_rows,
    }


def _wordfreq_score(rank, valid_count, zipf_value, min_zipf, max_zipf):
    if max_zipf > min_zipf:
        zipf_fraction = (max_zipf - zipf_value) / (max_zipf - min_zipf)
    else:
        zipf_fraction = rank / max(1, valid_count - 1)
    rank_fraction = rank / max(1, valid_count - 1)
    return clamp_score(1000 + zipf_fraction * 6500 + rank_fraction * 500)


def load_wordfreq_words(deny_words=None):
    deny_words = deny_words or set()
    if WORD_FREQ_TOP_N < 50000:
        raise ValueError("WORD_FREQ_TOP_N must be at least 50000")

    request_n = max(WORD_FREQ_TOP_N, int(WORD_FREQ_TOP_N * 1.15) + 1000)
    raw_words = []
    valid_words = []
    seen = set()
    excluded = defaultdict(int)

    while True:
        raw_words = top_n_list("en", request_n, wordlist=WORD_FREQ_WORDLIST, ascii_only=True)
        valid_words.clear()
        seen.clear()
        excluded.clear()

        for raw_word in raw_words:
            normalized = normalize_wordfreq_word(raw_word)
            if not normalized:
                excluded["invalid_format"] += 1
                continue
            if normalized in deny_words:
                excluded["deny_words"] += 1
                continue
            if normalized in seen:
                excluded["duplicate_after_normalization"] += 1
                continue
            seen.add(normalized)
            valid_words.append(normalized)
            if len(valid_words) >= WORD_FREQ_TOP_N:
                break

        if len(valid_words) >= WORD_FREQ_TOP_N or len(raw_words) < request_n:
            break
        request_n = int(request_n * 1.2) + 1000

    if len(valid_words) < 50000:
        raise RuntimeError(f"wordfreq valid word count is below 50000: {len(valid_words)}")

    zipfs = [zipf_frequency(word, "en", wordlist=WORD_FREQ_WORDLIST) for word in valid_words]
    min_zipf = min(zipfs) if zipfs else 0.0
    max_zipf = max(zipfs) if zipfs else 0.0

    entries = [
        Entry(
            input_word=word,
            output_word=word,
            pos_tag="X",
            score=_wordfreq_score(rank, len(valid_words), zipf_value, min_zipf, max_zipf),
            source="wordfreq",
        )
        for rank, (word, zipf_value) in enumerate(zip(valid_words, zipfs))
    ]

    print(
        "Loaded wordfreq words: "
        f"requested_top_n={WORD_FREQ_TOP_N}, read_words={len(raw_words)}, "
        f"valid_words={len(entries)}, excluded_words={sum(excluded.values())}"
    )
    return entries, {
        "requested_top_n": WORD_FREQ_TOP_N,
        "wordlist": WORD_FREQ_WORDLIST,
        "read_words": len(raw_words),
        "valid_words": len(entries),
        "excluded_words": sum(excluded.values()),
        "excluded_by_reason": dict(excluded),
        "min_zipf": min_zipf,
        "max_zipf": max_zipf,
    }


def load_scowl_words_optional(deny_words=None):
    deny_words = deny_words or set()
    if not SCOWL_WORDS_PATH.exists():
        print(f"WARNING: optional SCOWL wordlist not found: {SCOWL_WORDS_PATH}; continuing without SCOWL.")
        return [], {
            "path": str(SCOWL_WORDS_PATH),
            "present": False,
            "read_words": 0,
            "valid_words": 0,
            "excluded_words": 0,
        }

    entries = []
    seen = set()
    excluded = defaultdict(int)
    read_words = 0
    with SCOWL_WORDS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            read_words += 1
            normalized = normalize_scowl_word(line)
            if not normalized:
                excluded["invalid_format"] += 1
                continue
            if normalized in deny_words:
                excluded["deny_words"] += 1
                continue
            if normalized in seen:
                excluded["duplicate_after_normalization"] += 1
                continue
            seen.add(normalized)
            entries.append(Entry(normalized, normalized, "X", 9000, "scowl"))

    print(
        "Loaded SCOWL words: "
        f"read_words={read_words}, valid_words={len(entries)}, excluded_words={sum(excluded.values())}"
    )
    return entries, {
        "path": str(SCOWL_WORDS_PATH),
        "present": True,
        "read_words": read_words,
        "valid_words": len(entries),
        "excluded_words": sum(excluded.values()),
        "excluded_by_reason": dict(excluded),
    }


def load_corpus_words(deny_words=None):
    deny_words = deny_words or set()
    if not ENABLE_CORPUS:
        print("WARNING: corpus layer disabled; set ENABLE_CORPUS=1 to stream configured datasets.")
        return [], {
            "enabled": False,
            "read_documents": 0,
            "valid_words": 0,
            "excluded_words": 0,
            "datasets": DATASET_CONFIGS,
        }

    from datasets import load_dataset
    import spacy

    print("Loading spaCy model for corpus layer...")
    nlp = spacy.load("en_core_web_sm", disable=["parser"])
    print("spaCy model loaded.")

    unigram_counts = Counter()
    word_details = {}
    read_documents = 0
    excluded = defaultdict(int)

    for config in DATASET_CONFIGS:
        dataset_name = config["name"]
        print(f"Processing corpus dataset: {dataset_name} (split: {config['split']})")
        dataset = load_dataset(
            dataset_name,
            config.get("config"),
            split=config["split"],
            streaming=True,
            trust_remote_code=True,
        )
        texts_generator = (item[config["column"]] for item in dataset)

        for doc in tqdm(nlp.pipe(texts_generator, batch_size=500), desc=f"Processing {dataset_name}"):
            read_documents += 1
            words_to_count = []
            processed_entity_tokens = set()

            for ent in doc.ents:
                if ent.label_ in ENTITY_LABELS:
                    root_token = ent.root
                    if root_token.is_alpha:
                        lw = root_token.lower_
                        normalized = normalize_wordfreq_word(lw)
                        if normalized and normalized not in deny_words:
                            words_to_count.append(normalized)
                            word_details[normalized] = {"original": root_token.text, "pos": ent.label_}
                            processed_entity_tokens.add(root_token)
                        else:
                            excluded["invalid_or_denied_entity"] += 1

            for token in doc:
                if token in processed_entity_tokens:
                    continue
                if token.is_alpha:
                    lw = token.lower_
                    normalized = normalize_wordfreq_word(lw)
                    if normalized and normalized not in deny_words:
                        words_to_count.append(normalized)
                        word_details.setdefault(normalized, {"original": normalized, "pos": token.pos_})
                    else:
                        excluded["invalid_or_denied_token"] += 1

            unigram_counts.update(words_to_count)
            if CORPUS_MAX_DOCS and read_documents >= CORPUS_MAX_DOCS:
                break

    entries = []
    total_unigrams = sum(unigram_counts.values())
    if total_unigrams:
        costs = {
            word: -math.log(count / total_unigrams)
            for word, count in unigram_counts.items()
            if count > 0 and word in word_details
        }
        min_cost = min(costs.values())
        max_cost = max(costs.values())
        for word, cost in costs.items():
            if max_cost > min_cost:
                fraction = (cost - min_cost) / (max_cost - min_cost)
            else:
                fraction = 0.0
            score = clamp_score(9000 + fraction * 21000)
            details = word_details[word]
            entries.append(Entry(word, details["original"], details["pos"], score, "corpus"))

    print(
        "Loaded corpus words: "
        f"read_documents={read_documents}, valid_words={len(entries)}, excluded_words={sum(excluded.values())}"
    )
    return entries, {
        "enabled": True,
        "read_documents": read_documents,
        "valid_words": len(entries),
        "excluded_words": sum(excluded.values()),
        "excluded_by_reason": dict(excluded),
        "total_unigrams": int(total_unigrams),
        "datasets": DATASET_CONFIGS,
    }


def merge_entries(entries):
    merged = {}
    source_counts_before_merge = Counter()
    source_counts_after_merge = Counter()
    rejected = 0

    for entry in entries:
        source_counts_before_merge[entry.source] += 1
        if not validate_entry(entry):
            rejected += 1
            continue
        key = f"{entry.input_word}\t{entry.output_word}"
        existing = merged.get(key)
        if existing is None or entry.score < existing.score:
            merged[key] = entry

    final_entries = sorted(merged.values(), key=lambda item: (item.score, item.input_word, item.output_word))
    for entry in final_entries:
        source_counts_after_merge[entry.source] += 1

    return final_entries, {
        "input_entries": len(entries),
        "output_entries": len(final_entries),
        "unique_input_words": len({entry.input_word for entry in final_entries}),
        "rejected_entries": rejected,
        "source_counts_before_merge": dict(source_counts_before_merge),
        "source_counts_after_merge": dict(source_counts_after_merge),
    }


def write_output(entries, output_filename=OUTPUT_FILENAME):
    with Path(output_filename).open("w", encoding="utf-8", newline="") as f:
        f.write("input_word\toutput_word\tpos_tag\tscore\n")
        for entry in entries:
            f.write(f"{entry.input_word}\t{entry.output_word}\t{entry.pos_tag}\t{entry.score}\n")
    print(f"Saved dictionary TSV: {output_filename} ({len(entries)} entries)")


def write_source_summary(summary):
    with Path(SUMMARY_FILENAME).open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Saved source summary: {SUMMARY_FILENAME}")


def zip_output(output_filename=OUTPUT_FILENAME):
    files_to_zip = [Path(output_filename)]
    for path in [Path(SUMMARY_FILENAME), Path("NOTICE.md")]:
        if path.exists():
            files_to_zip.append(path)
    licenses_dir = Path("LICENSES")
    if licenses_dir.exists():
        files_to_zip.extend(sorted(path for path in licenses_dir.iterdir() if path.is_file()))

    with zipfile.ZipFile(ZIP_FILENAME, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files_to_zip:
            zf.write(path, arcname=str(path))
    print(f"Saved ZIP: {ZIP_FILENAME}")


def main():
    summary = {
        "configuration": {
            "word_freq_top_n": WORD_FREQ_TOP_N,
            "word_freq_wordlist": WORD_FREQ_WORDLIST,
            "enable_corpus": ENABLE_CORPUS,
            "corpus_max_docs": CORPUS_MAX_DOCS,
            "output_filename": OUTPUT_FILENAME,
            "zip_filename": ZIP_FILENAME,
        },
        "sources": {},
    }

    deny_words, deny_summary = load_deny_words()
    summary["sources"]["deny_words"] = deny_summary

    all_entries = []
    manual_summaries = {}
    for path in MANUAL_TSV_PATHS:
        entries, manual_summary = load_manual_tsv(path)
        all_entries.extend(entries)
        manual_summaries[path.stem] = manual_summary
    summary["sources"]["manual"] = manual_summaries

    wordfreq_entries, wordfreq_summary = load_wordfreq_words(deny_words)
    all_entries.extend(wordfreq_entries)
    summary["sources"]["wordfreq"] = wordfreq_summary

    scowl_entries, scowl_summary = load_scowl_words_optional(deny_words)
    if scowl_summary.get("present"):
        wordfreq_word_set = {entry.input_word for entry in wordfreq_entries}
        scowl_word_set = {entry.input_word for entry in scowl_entries}
        scowl_summary["coverage_report"] = {
            "wordfreq_words_present_in_scowl": len(wordfreq_word_set & scowl_word_set),
            "wordfreq_words_absent_from_scowl": len(wordfreq_word_set - scowl_word_set),
            "scowl_words_absent_from_wordfreq": len(scowl_word_set - wordfreq_word_set),
        }
    all_entries.extend(scowl_entries)
    summary["sources"]["scowl"] = scowl_summary

    corpus_entries, corpus_summary = load_corpus_words(deny_words)
    all_entries.extend(corpus_entries)
    summary["sources"]["corpus"] = corpus_summary

    merged_entries, merge_summary = merge_entries(all_entries)
    summary["merge"] = merge_summary
    summary["sample_entries"] = [asdict(entry) for entry in merged_entries[:20]]

    if summary["sources"]["wordfreq"]["valid_words"] < 50000:
        raise RuntimeError("wordfreq layer valid_words is below 50000")
    if summary["merge"]["unique_input_words"] < 50000:
        raise RuntimeError("final unique input_word count is below 50000")

    write_output(merged_entries, OUTPUT_FILENAME)
    write_source_summary(summary)
    zip_output(OUTPUT_FILENAME)


if __name__ == "__main__":
    main()
