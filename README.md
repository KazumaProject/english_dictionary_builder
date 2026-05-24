# English IME dictionary builder

This project generates English IME dictionary artifacts for candidate ranking.
Unigrams and phrases are intentionally separate artifacts:

```text
1-grams_score_cost_pos_combined_with_ner.txt
1-grams_score_cost_pos_combined_with_ner.zip
english_phrases.tsv
english_phrases.zip
dictionary_source_summary.json
```

The unigram TSV columns are:

```text
input_word	output_word	pos_tag	score
```

The phrase TSV columns are:

```text
input_phrase	output_phrase	pos_tag	score
```

Lower `score` values rank higher.

## Why stop words are included

English IMEs need function words and stop words as first-class candidates.
Words such as `because`, `the`, `and`, `you`, `I`, `is`, `are`, `with`, `from`,
and `about` are common typing targets, not noise. The generator therefore does
not filter out stop words.

## Data sources

### wordfreq

`wordfreq` is the main unigram vocabulary and frequency-ranking source. It
combines multiple corpora and exposes English top-word lists plus Zipf
frequencies, which are suitable for stable IME cost assignment.

Configuration:

```bash
WORD_FREQ_TOP_N=100000 python main.py
```

`WORD_FREQ_TOP_N` defaults to `100000`. The generator fails if the wordfreq
layer yields fewer than `50000` valid words. `100000` or more unique input words
is treated as standard coverage, and `200000` or more is treated as high
coverage by the verifier.

The wordfreq layer:

- reads English words using `top_n_list("en", ..., wordlist="large", ascii_only=True)`
- scores them using `zipf_frequency(..., "en", wordlist="large")`
- skips apostrophe forms, URLs, HTML fragments, empty values, numbers, symbols,
  mixed alphanumeric tokens, and words outside length `1..32`
- allows one-letter words only when they are `a` or `i`
- applies `data/manual/deny_words.tsv`
- writes lowercase `input_word` and lowercase `output_word`

Contractions such as `I'm` and `don't` are handled in
`data/manual/contractions.tsv`, not by the wordfreq layer.

### Manual unigrams

Manual TSV files are explicit project-authored data and are not filtered by
`deny_words.tsv`. They are validated strictly: malformed rows, empty fields,
invalid lowercase input words, control characters, or scores outside
`0..32767` fail the build.

Current manual unigram sources:

- `data/manual/basic_words.tsv`: high-priority function words, pronouns,
  auxiliaries, prepositions, and everyday adverbs.
- `data/manual/contractions.tsv`: apostrophe outputs such as `im -> I'm` and
  `dont -> don't`.
- `data/manual/spelling_variants.tsv`: common US, Canadian, and British spelling
  variants.
- `data/manual/tech_common_words.tsv`: compatibility source for existing tech
  and product terms.
- `data/manual/common_app_words.tsv`: app and account terms.
- `data/manual/mobile_words.tsv`: mobile device and UI terms.
- `data/manual/android_ime_words.tsv`: Android IME, keyboard, and Japanese input
  terms.
- `data/manual/software_dev_words.tsv`: developer workflow and file format
  terms.
- `data/manual/product_brand_words.tsv`: product and brand display spellings.
- `data/manual/deny_words.tsv`: project-controlled exclusions for wordfreq,
  corpus, SCOWL, and generated inflections. Manual TSV entries are not filtered
  by this deny list.

### Inflections

Inflections are generated only for lemmas explicitly listed in:

```text
data/manual/inflection_lemmas.tsv
data/manual/inflection_overrides.tsv
```

The generator does not mechanically add `s`, `ed`, or `ing` to every wordfreq
word. Regular generation is limited to these rules:

- `VERB`: `third_person_s`, `past_ed`, `present_participle_ing`
- `NOUN`: `plural_s`, `plural_es`, `plural_y_to_ies`
- `ADJ`: `comparative_er`, `superlative_est`

Irregular forms must be listed in `inflection_overrides.tsv`. Generated forms
are accepted only when the form is lowercase ASCII alphabetic, not present in
`deny_words.tsv`, and validated by wordfreq or optional SCOWL. When SCOWL is
absent, wordfreq is the only validation source. Inflection candidate, accepted,
rejected, validation-source, and override counts are recorded in
`dictionary_source_summary.json`.

### Phrases and bigrams

Phrase/bigram entries are not mixed into the unigram TSV. They are built as a
separate artifact:

```text
english_phrases.tsv
english_phrases.zip
```

At present, phrase generation uses only manual phrase sources:

- `data/manual/phrases_common.tsv`
- `data/manual/phrases_mobile_ime.tsv`
- `data/manual/phrases_tech.tsv`

`ENABLE_CORPUS_BIGRAM` is reserved in the summary configuration and defaults to
false. Corpus bigram extraction is not implemented in this build path so GitHub
Actions output stays deterministic.

### Optional SCOWL / English Speller Database

SCOWL is optional. To use it, place a plain one-word-per-line word list at:

```text
external_sources/scowl/words.txt
```

If the file is absent, generation continues with a warning and SCOWL is recorded
as an unused source. SCOWL is used as optional spell validation, optional
enrichment, and coverage-report source. wordfreq words are not automatically
removed just because they are absent from SCOWL.

When adding SCOWL locally, keep the upstream license or copyright notice at:

```text
external_sources/scowl/LICENSE_OR_COPYRIGHT.txt
```

SCOWL / English Speller Database project page: http://wordlist.aspell.net/

### Optional corpus layer

The previous Wikitext/spaCy corpus processing remains available but is disabled
by default so the wordfreq + manual build can complete reliably:

```bash
ENABLE_CORPUS=1 python main.py
```

For smoke tests:

```bash
ENABLE_CORPUS=1 CORPUS_MAX_DOCS=1000 python main.py
```

The corpus layer does not filter `token.is_stop`; it only requires alphabetic
tokens and then applies the same normalization and deny-word rules as the
wordfreq layer.

## Scoring and merge rules

- Manual important words generally stay in the `700..2200` range.
- Tech, mobile, IME, and product-name manual entries generally use
  `1500..3500`.
- wordfreq scores keep the existing Zipf-frequency and rank logic, with common
  words receiving lower costs.
- Inflection scores are slightly lower priority than their lemma scores.
- Common phrases generally use `1200..2200`; tech phrases generally use
  `1300..3000`.
- Corpus scores keep the existing `-log(count / total)` idea and map corpus
  words into a lower-priority integer range.
- Scores are clamped to `0..32767` for generated layers.
- Manual TSV scores must already be in `0..32767`.
- Unigram merge key is `input_word + "\t" + output_word`.
- Phrase merge key is `input_phrase + "\t" + output_phrase`.
- Duplicate keys keep the lower score.

## Build

Install dependencies:

```bash
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

The spaCy model is only required when `ENABLE_CORPUS=1`.

Generate the artifacts:

```bash
python main.py
```

Verify generated TSVs, ZIP contents, required pairs, source summary structure,
and minimum unigram coverage:

```bash
python verify_dictionary_output.py
```

## GitHub Actions release flow

`.github/workflows/create-release.yml` runs on tag pushes matching `v*.*.*`.
The workflow:

- installs dependencies with `pip install -r requirements.txt`
- runs `python main.py`
- runs `python verify_dictionary_output.py`
- checks ZIP contents before upload
- creates a GitHub Release with the generated artifacts and license files

Release upload targets include:

```text
1-grams_score_cost_pos_combined_with_ner.zip
english_phrases.zip
dictionary_source_summary.json
NOTICE.md
LICENSE
LICENSES/**
```

## Outputs

- `1-grams_score_cost_pos_combined_with_ner.txt`: final unigram dictionary TSV.
- `1-grams_score_cost_pos_combined_with_ner.zip`: ZIP package containing the
  unigram TSV, summary, and notice/license files when present.
- `english_phrases.tsv`: final phrase dictionary TSV.
- `english_phrases.zip`: ZIP package containing the phrase TSV, summary, and
  notice/license files when present.
- `dictionary_source_summary.json`: configuration, source counts, inflection
  validation counts, SCOWL status, corpus status, and unigram/phrase merge
  counts.

## Licenses and notices

Project code is licensed under Apache License 2.0. See `LICENSE`.

Manual TSV files in this repository are project-authored data.

Generated dictionary data includes wordfreq-derived data. The upstream
`wordfreq` repository states that the code is redistributable under Apache
License 2.0 and that it includes data files redistributable under Creative
Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0), with additional
source notices documented upstream.

Relevant local notice files:

- `NOTICE.md`
- `LICENSES/wordfreq-NOTICE.md`
- `LICENSES/CC-BY-SA-4.0.txt`

When distributing generated dictionary data, include the generated ZIPs together
with these notices and licenses.
