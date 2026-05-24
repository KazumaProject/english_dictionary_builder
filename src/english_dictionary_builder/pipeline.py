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
PHRASE_OUTPUT_FILENAME = "english_phrases.tsv"
PHRASE_ZIP_FILENAME = "english_phrases.zip"
SUMMARY_FILENAME = "dictionary_source_summary.json"

WORD_FREQ_TOP_N = int(os.getenv("WORD_FREQ_TOP_N", "100000"))
WORD_FREQ_WORDLIST = os.getenv("WORD_FREQ_WORDLIST", "large")
ENABLE_CORPUS = os.getenv("ENABLE_CORPUS", "0").lower() in {"1", "true", "yes", "on"}
ENABLE_CORPUS_BIGRAM = os.getenv("ENABLE_CORPUS_BIGRAM", "0").lower() in {"1", "true", "yes", "on"}
CORPUS_MAX_DOCS = int(os.getenv("CORPUS_MAX_DOCS", "0"))

DATASET_CONFIGS = [
    {"name": "wikitext", "config": "wikitext-103-v1", "split": "train", "column": "text"},
]

MANUAL_TSV_PATHS = [
    Path("data/manual/basic_words.tsv"),
    Path("data/manual/contractions.tsv"),
    Path("data/manual/spelling_variants.tsv"),
    Path("data/manual/tech_common_words.tsv"),
    Path("data/manual/common_app_words.tsv"),
    Path("data/manual/mobile_words.tsv"),
    Path("data/manual/android_ime_words.tsv"),
    Path("data/manual/software_dev_words.tsv"),
    Path("data/manual/product_brand_words.tsv"),
]

INFLECTION_LEMMAS_PATH = Path("data/manual/inflection_lemmas.tsv")
INFLECTION_OVERRIDES_PATH = Path("data/manual/inflection_overrides.tsv")

MANUAL_PHRASE_TSV_PATHS = [
    Path("data/manual/phrases_common.tsv"),
    Path("data/manual/phrases_mobile_ime.tsv"),
    Path("data/manual/phrases_tech.tsv"),
]

DENY_WORDS_PATH = Path("data/manual/deny_words.tsv")
SCOWL_WORDS_PATH = Path("external_sources/scowl/words.txt")

WORD_RE = re.compile(r"^[a-z]+$")
ENTITY_LABELS = {"GPE", "PERSON", "ORG", "PRODUCT", "LOC", "FAC", "EVENT", "WORK_OF_ART"}
VOWELS = set("aeiou")


@dataclass(frozen=True)
class Entry:
    input_word: str
    output_word: str
    pos_tag: str
    score: int
    source: str


@dataclass(frozen=True)
class PhraseEntry:
    input_phrase: str
    output_phrase: str
    pos_tag: str
    score: int
    source: str


@dataclass(frozen=True)
class InflectionLemma:
    lemma: str
    pos_tag: str
    base_score: int


@dataclass(frozen=True)
class InflectionOverride:
    lemma: str
    form: str
    pos_tag: str
    inflection_type: str
    score: int


def clamp_score(score):
    return max(0, min(32767, int(round(score))))


def parse_score(score_text, path, line_number):
    try:
        score = int(score_text)
    except ValueError as exc:
        raise ValueError(f"{path}:{line_number} score must be an integer: {score_text!r}") from exc
    if not 0 <= score <= 32767:
        raise ValueError(f"{path}:{line_number} score must be in 0..32767: {score}")
    return score


def is_ascii_alpha_lower(value):
    return bool(value and WORD_RE.fullmatch(value))


def is_allowed_input_word(word):
    if not word or not WORD_RE.fullmatch(word):
        return False
    if not 1 <= len(word) <= 32:
        return False
    if len(word) == 1 and word not in {"a", "i"}:
        return False
    return True


def is_allowed_phrase_input(phrase):
    return is_ascii_alpha_lower(phrase)


def validate_display_field(value):
    return bool(value) and "\t" not in value and "\n" not in value and "\r" not in value


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
    if not is_allowed_input_word(entry.input_word):
        return False
    if not validate_display_field(entry.output_word):
        return False
    if not validate_display_field(entry.pos_tag):
        return False
    return 0 <= int(entry.score) <= 32767


def validate_phrase_entry(entry):
    if not is_allowed_phrase_input(entry.input_phrase):
        return False
    if not validate_display_field(entry.output_phrase):
        return False
    if not validate_display_field(entry.pos_tag):
        return False
    return 0 <= int(entry.score) <= 32767


def require_existing_paths(paths):
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Required manual TSV file(s) missing: " + ", ".join(missing))


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
            if input_word != normalized_input or not is_allowed_input_word(normalized_input):
                raise ValueError(f"{path}:{line_number} has invalid input_word: {input_word!r}")
            if not validate_display_field(output_word.strip()):
                raise ValueError(f"{path}:{line_number} has invalid output_word")
            if not validate_display_field(pos_tag.strip()):
                raise ValueError(f"{path}:{line_number} has invalid pos_tag")
            entry = Entry(
                input_word=normalized_input,
                output_word=output_word.strip(),
                pos_tag=pos_tag.strip(),
                score=parse_score(score_text, path, line_number),
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


def load_inflection_lemmas(path):
    lemmas = []
    read_rows = 0
    excluded_rows = 0

    with Path(path).open("r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        if header != ["lemma", "pos_tag", "base_score"]:
            raise ValueError(f"{path} must have header: lemma\\tpos_tag\\tbase_score")

        for line_number, line in enumerate(f, start=2):
            read_rows += 1
            stripped = line.rstrip("\n")
            if not stripped:
                excluded_rows += 1
                continue
            parts = stripped.split("\t")
            if len(parts) != 3:
                raise ValueError(f"{path}:{line_number} must have 3 tab-separated columns")
            lemma, pos_tag, base_score = parts
            normalized_lemma = lemma.strip().lower()
            if lemma != normalized_lemma or not is_allowed_input_word(normalized_lemma):
                raise ValueError(f"{path}:{line_number} has invalid lemma: {lemma!r}")
            if pos_tag not in {"VERB", "NOUN", "ADJ"}:
                raise ValueError(f"{path}:{line_number} has unsupported pos_tag: {pos_tag!r}")
            lemmas.append(InflectionLemma(normalized_lemma, pos_tag, parse_score(base_score, path, line_number)))

    print(f"Loaded inflection lemmas {path}: read_rows={read_rows}, valid_rows={len(lemmas)}")
    return lemmas, {
        "path": str(path),
        "read_rows": read_rows,
        "valid_rows": len(lemmas),
        "excluded_rows": excluded_rows,
    }


def load_inflection_overrides(path):
    overrides = []
    read_rows = 0
    excluded_rows = 0

    with Path(path).open("r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        if header != ["lemma", "form", "pos_tag", "inflection_type", "score"]:
            raise ValueError(f"{path} must have header: lemma\\tform\\tpos_tag\\tinflection_type\\tscore")

        for line_number, line in enumerate(f, start=2):
            read_rows += 1
            stripped = line.rstrip("\n")
            if not stripped:
                excluded_rows += 1
                continue
            parts = stripped.split("\t")
            if len(parts) != 5:
                raise ValueError(f"{path}:{line_number} must have 5 tab-separated columns")
            lemma, form, pos_tag, inflection_type, score_text = parts
            normalized_lemma = lemma.strip().lower()
            normalized_form = form.strip().lower()
            if lemma != normalized_lemma or not is_allowed_input_word(normalized_lemma):
                raise ValueError(f"{path}:{line_number} has invalid lemma: {lemma!r}")
            if form != normalized_form or not is_allowed_input_word(normalized_form):
                raise ValueError(f"{path}:{line_number} has invalid form: {form!r}")
            if pos_tag not in {"VERB", "NOUN", "ADJ"}:
                raise ValueError(f"{path}:{line_number} has unsupported pos_tag: {pos_tag!r}")
            if not validate_display_field(inflection_type.strip()):
                raise ValueError(f"{path}:{line_number} has invalid inflection_type")
            overrides.append(
                InflectionOverride(
                    normalized_lemma,
                    normalized_form,
                    pos_tag,
                    inflection_type.strip(),
                    parse_score(score_text, path, line_number),
                )
            )

    print(f"Loaded inflection overrides {path}: read_rows={read_rows}, valid_rows={len(overrides)}")
    return overrides, {
        "path": str(path),
        "read_rows": read_rows,
        "valid_rows": len(overrides),
        "excluded_rows": excluded_rows,
    }


def load_manual_phrase_tsv(path):
    entries = []
    read_rows = 0
    excluded_rows = 0

    with Path(path).open("r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        if header != ["input_phrase", "output_phrase", "pos_tag", "score"]:
            raise ValueError(f"{path} must have header: input_phrase\\toutput_phrase\\tpos_tag\\tscore")

        for line_number, line in enumerate(f, start=2):
            read_rows += 1
            stripped = line.rstrip("\n")
            if not stripped:
                excluded_rows += 1
                continue
            parts = stripped.split("\t")
            if len(parts) != 4:
                raise ValueError(f"{path}:{line_number} must have 4 tab-separated columns")
            input_phrase, output_phrase, pos_tag, score_text = parts
            normalized_input = input_phrase.strip().lower()
            if input_phrase != normalized_input or not is_allowed_phrase_input(normalized_input):
                raise ValueError(f"{path}:{line_number} has invalid input_phrase: {input_phrase!r}")
            if not validate_display_field(output_phrase.strip()):
                raise ValueError(f"{path}:{line_number} has invalid output_phrase")
            if not validate_display_field(pos_tag.strip()):
                raise ValueError(f"{path}:{line_number} has invalid pos_tag")
            entry = PhraseEntry(
                input_phrase=normalized_input,
                output_phrase=output_phrase.strip(),
                pos_tag=pos_tag.strip(),
                score=parse_score(score_text, path, line_number),
                source=f"manual_phrase:{Path(path).stem}",
            )
            if not validate_phrase_entry(entry):
                raise ValueError(f"{path}:{line_number} has an invalid phrase entry")
            entries.append(entry)

    print(f"Loaded manual phrase TSV {path}: read_rows={read_rows}, valid_rows={len(entries)}")
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


def corpus_bigram_summary():
    return {
        "enabled": False,
        "requested": ENABLE_CORPUS_BIGRAM,
        "valid_phrases": 0,
        "note": "Reserved for future deterministic corpus bigram extraction; manual phrase TSVs are the only active phrase source.",
    }


def has_consonant_before_y(word):
    return len(word) >= 2 and word.endswith("y") and word[-2] not in VOWELS


def regular_inflection_candidates(lemma):
    candidates = []
    word = lemma.lemma

    if lemma.pos_tag == "VERB":
        candidates.append(("third_person_s", f"{word}s", clamp_score(lemma.base_score + 250)))
        candidates.append(("past_ed", f"{word}d" if word.endswith("e") else f"{word}ed", clamp_score(lemma.base_score + 350)))
        if word.endswith("ie"):
            ing_form = f"{word[:-2]}ying"
        elif word.endswith("e") and not word.endswith(("ee", "ye")):
            ing_form = f"{word[:-1]}ing"
        else:
            ing_form = f"{word}ing"
        candidates.append(("present_participle_ing", ing_form, clamp_score(lemma.base_score + 350)))
    elif lemma.pos_tag == "NOUN":
        candidates.append(("plural_s", f"{word}s", clamp_score(lemma.base_score + 250)))
        if word.endswith(("s", "x", "z", "ch", "sh")):
            candidates.append(("plural_es", f"{word}es", clamp_score(lemma.base_score + 300)))
        if has_consonant_before_y(word):
            candidates.append(("plural_y_to_ies", f"{word[:-1]}ies", clamp_score(lemma.base_score + 300)))
    elif lemma.pos_tag == "ADJ":
        er_form = f"{word}r" if word.endswith("e") else f"{word}er"
        est_form = f"{word}st" if word.endswith("e") else f"{word}est"
        candidates.append(("comparative_er", er_form, clamp_score(lemma.base_score + 300)))
        candidates.append(("superlative_est", est_form, clamp_score(lemma.base_score + 350)))

    return candidates


def validation_sources_for_form(form, wordfreq_valid_words, scowl_valid_words):
    sources = []
    if form in wordfreq_valid_words:
        sources.append("wordfreq")
    if form in scowl_valid_words:
        sources.append("scowl")
    return sources


def build_inflection_entries(lemmas, overrides, deny_words, wordfreq_entries, scowl_entries):
    wordfreq_valid_words = {entry.input_word for entry in wordfreq_entries}
    scowl_valid_words = {entry.input_word for entry in scowl_entries}
    generated_candidates = []
    rejected_by_reason = Counter()
    accepted_source_counts = Counter()
    accepted_entry_source_counts = Counter()
    accepted_samples = []
    entries = []
    seen_candidate_keys = set()

    for lemma in lemmas:
        for inflection_type, form, score in regular_inflection_candidates(lemma):
            generated_candidates.append(
                {
                    "lemma": lemma.lemma,
                    "form": form,
                    "pos_tag": lemma.pos_tag,
                    "inflection_type": inflection_type,
                    "score": score,
                    "entry_source": "regular",
                }
            )

    for override in overrides:
        generated_candidates.append(
            {
                "lemma": override.lemma,
                "form": override.form,
                "pos_tag": override.pos_tag,
                "inflection_type": override.inflection_type,
                "score": override.score,
                "entry_source": "override",
            }
        )

    for candidate in generated_candidates:
        form = candidate["form"]
        candidate_key = (candidate["lemma"], form, candidate["pos_tag"], candidate["inflection_type"])
        if candidate_key in seen_candidate_keys:
            rejected_by_reason["duplicate_candidate"] += 1
            continue
        seen_candidate_keys.add(candidate_key)

        if not is_allowed_input_word(form):
            rejected_by_reason["invalid_form"] += 1
            continue
        if form in deny_words:
            rejected_by_reason["deny_words"] += 1
            continue
        validation_sources = validation_sources_for_form(form, wordfreq_valid_words, scowl_valid_words)
        if not validation_sources:
            rejected_by_reason["not_in_validation_source"] += 1
            continue

        score = clamp_score(candidate["score"])
        entry = Entry(
            input_word=form,
            output_word=form,
            pos_tag=candidate["pos_tag"],
            score=score,
            source=f"inflection:{candidate['entry_source']}:{candidate['inflection_type']}",
        )
        if not validate_entry(entry):
            rejected_by_reason["invalid_entry"] += 1
            continue

        entries.append(entry)
        accepted_entry_source_counts[candidate["entry_source"]] += 1
        for validation_source in validation_sources:
            accepted_source_counts[validation_source] += 1
        if len(accepted_samples) < 20:
            accepted_samples.append(
                {
                    "lemma": candidate["lemma"],
                    "form": form,
                    "pos_tag": candidate["pos_tag"],
                    "inflection_type": candidate["inflection_type"],
                    "entry_source": candidate["entry_source"],
                    "validation_sources": validation_sources,
                    "score": score,
                }
            )

    summary = {
        "enabled": True,
        "lemma_row_count": len(lemmas),
        "override_row_count": len(overrides),
        "generated_candidate_count": len(generated_candidates),
        "accepted_count": len(entries),
        "rejected_count": sum(rejected_by_reason.values()),
        "rejected_by_reason": dict(rejected_by_reason),
        "accepted_source_counts": dict(accepted_source_counts),
        "accepted_entry_source_counts": dict(accepted_entry_source_counts),
        "validation_sources": ["wordfreq"] + (["scowl"] if scowl_valid_words else []),
        "accepted_samples": accepted_samples,
    }
    print(
        "Built inflections: "
        f"candidates={summary['generated_candidate_count']}, accepted={summary['accepted_count']}, "
        f"rejected={summary['rejected_count']}"
    )
    return entries, summary


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


def merge_phrase_entries(entries):
    merged = {}
    source_counts_before_merge = Counter()
    source_counts_after_merge = Counter()
    rejected = 0

    for entry in entries:
        source_counts_before_merge[entry.source] += 1
        if not validate_phrase_entry(entry):
            rejected += 1
            continue
        key = f"{entry.input_phrase}\t{entry.output_phrase}"
        existing = merged.get(key)
        if existing is None or entry.score < existing.score:
            merged[key] = entry

    final_entries = sorted(merged.values(), key=lambda item: (item.score, item.input_phrase, item.output_phrase))
    for entry in final_entries:
        source_counts_after_merge[entry.source] += 1

    return final_entries, {
        "input_entries": len(entries),
        "output_entries": len(final_entries),
        "unique_input_phrases": len({entry.input_phrase for entry in final_entries}),
        "rejected_entries": rejected,
        "source_counts_before_merge": dict(source_counts_before_merge),
        "source_counts_after_merge": dict(source_counts_after_merge),
    }


def write_unigram_output(entries, output_filename=OUTPUT_FILENAME):
    with Path(output_filename).open("w", encoding="utf-8", newline="") as f:
        f.write("input_word\toutput_word\tpos_tag\tscore\n")
        for entry in entries:
            f.write(f"{entry.input_word}\t{entry.output_word}\t{entry.pos_tag}\t{entry.score}\n")
    print(f"Saved dictionary TSV: {output_filename} ({len(entries)} entries)")


def write_phrase_output(entries, output_filename=PHRASE_OUTPUT_FILENAME):
    with Path(output_filename).open("w", encoding="utf-8", newline="") as f:
        f.write("input_phrase\toutput_phrase\tpos_tag\tscore\n")
        for entry in entries:
            f.write(f"{entry.input_phrase}\t{entry.output_phrase}\t{entry.pos_tag}\t{entry.score}\n")
    print(f"Saved phrase TSV: {output_filename} ({len(entries)} entries)")


def write_source_summary(summary):
    with Path(SUMMARY_FILENAME).open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Saved source summary: {SUMMARY_FILENAME}")


def license_notice_files():
    files = []
    for path in [Path("NOTICE.md")]:
        if path.exists():
            files.append(path)
    licenses_dir = Path("LICENSES")
    if licenses_dir.exists():
        files.extend(sorted(path for path in licenses_dir.rglob("*") if path.is_file()))
    return files


def zip_unigram_output(output_filename=OUTPUT_FILENAME):
    files_to_zip = [Path(output_filename), Path(SUMMARY_FILENAME)] + license_notice_files()

    with zipfile.ZipFile(ZIP_FILENAME, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files_to_zip:
            if path.exists():
                zf.write(path, arcname=str(path))
    print(f"Saved ZIP: {ZIP_FILENAME}")


def zip_phrase_output(output_filename=PHRASE_OUTPUT_FILENAME):
    files_to_zip = [Path(output_filename), Path(SUMMARY_FILENAME)] + license_notice_files()

    with zipfile.ZipFile(PHRASE_ZIP_FILENAME, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files_to_zip:
            if path.exists():
                zf.write(path, arcname=str(path))
    print(f"Saved ZIP: {PHRASE_ZIP_FILENAME}")


def main():
    require_existing_paths(MANUAL_TSV_PATHS + MANUAL_PHRASE_TSV_PATHS + [INFLECTION_LEMMAS_PATH, INFLECTION_OVERRIDES_PATH])

    summary = {
        "configuration": {
            "word_freq_top_n": WORD_FREQ_TOP_N,
            "word_freq_wordlist": WORD_FREQ_WORDLIST,
            "enable_corpus": ENABLE_CORPUS,
            "enable_corpus_bigram": ENABLE_CORPUS_BIGRAM,
            "corpus_max_docs": CORPUS_MAX_DOCS,
            "unigram_output_filename": OUTPUT_FILENAME,
            "unigram_zip_filename": ZIP_FILENAME,
            "phrase_output_filename": PHRASE_OUTPUT_FILENAME,
            "phrase_zip_filename": PHRASE_ZIP_FILENAME,
            "output_filename": OUTPUT_FILENAME,
            "zip_filename": ZIP_FILENAME,
        },
        "sources": {},
    }

    deny_words, deny_summary = load_deny_words()
    summary["sources"]["deny_words"] = deny_summary

    all_unigram_entries = []
    manual_summaries = {}
    for path in MANUAL_TSV_PATHS:
        entries, manual_summary = load_manual_tsv(path)
        all_unigram_entries.extend(entries)
        manual_summaries[path.stem] = manual_summary
    summary["sources"]["manual_unigrams"] = manual_summaries

    phrase_entries = []
    phrase_summaries = {}
    for path in MANUAL_PHRASE_TSV_PATHS:
        entries, phrase_summary = load_manual_phrase_tsv(path)
        phrase_entries.extend(entries)
        phrase_summaries[path.stem] = phrase_summary
    summary["sources"]["manual_phrases"] = phrase_summaries

    lemmas, lemma_summary = load_inflection_lemmas(INFLECTION_LEMMAS_PATH)
    overrides, override_summary = load_inflection_overrides(INFLECTION_OVERRIDES_PATH)
    summary["sources"]["manual_inflection_lemmas"] = lemma_summary
    summary["sources"]["manual_inflection_overrides"] = override_summary

    wordfreq_entries, wordfreq_summary = load_wordfreq_words(deny_words)
    all_unigram_entries.extend(wordfreq_entries)
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
    all_unigram_entries.extend(scowl_entries)
    summary["sources"]["scowl"] = scowl_summary

    inflection_entries, inflection_summary = build_inflection_entries(
        lemmas,
        overrides,
        deny_words,
        wordfreq_entries,
        scowl_entries,
    )
    all_unigram_entries.extend(inflection_entries)
    summary["sources"]["inflections"] = inflection_summary

    corpus_entries, corpus_summary = load_corpus_words(deny_words)
    all_unigram_entries.extend(corpus_entries)
    summary["sources"]["corpus"] = corpus_summary
    summary["sources"]["corpus_bigram"] = corpus_bigram_summary()

    merged_unigram_entries, unigram_merge_summary = merge_entries(all_unigram_entries)
    merged_phrase_entries, phrase_merge_summary = merge_phrase_entries(phrase_entries)
    summary["merge"] = {
        "unigram": unigram_merge_summary,
        "phrase": phrase_merge_summary,
        "source_counts_before_merge": unigram_merge_summary["source_counts_before_merge"],
        "source_counts_after_merge": unigram_merge_summary["source_counts_after_merge"],
    }
    summary["sample_unigram_entries"] = [asdict(entry) for entry in merged_unigram_entries[:20]]
    summary["sample_phrase_entries"] = [asdict(entry) for entry in merged_phrase_entries[:20]]

    if summary["sources"]["wordfreq"]["valid_words"] < 50000:
        raise RuntimeError("wordfreq layer valid_words is below 50000")
    if summary["merge"]["unigram"]["unique_input_words"] < 50000:
        raise RuntimeError("final unique input_word count is below 50000")

    write_unigram_output(merged_unigram_entries, OUTPUT_FILENAME)
    write_phrase_output(merged_phrase_entries, PHRASE_OUTPUT_FILENAME)
    write_source_summary(summary)
    zip_unigram_output(OUTPUT_FILENAME)
    zip_phrase_output(PHRASE_OUTPUT_FILENAME)


if __name__ == "__main__":
    main()
