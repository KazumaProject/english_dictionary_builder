# English IME one-gram dictionary builder

This project generates an English one-gram dictionary TSV for IME candidate
ranking. The main output is:

```text
1-grams_score_cost_pos_combined_with_ner.txt
1-grams_score_cost_pos_combined_with_ner.zip
dictionary_source_summary.json
```

The TSV columns are:

```text
input_word	output_word	pos_tag	score
```

Lower `score` values rank higher.

## Why stop words are included

English IMEs need function words and stop words as first-class candidates.
Words such as `because`, `the`, `and`, `you`, `I`, `is`, `are`, `with`, `from`,
and `about` are common typing targets, not noise. The generator therefore does
not filter out stop words.

## Data sources

### wordfreq

`wordfreq` is the main vocabulary and frequency-ranking source. It combines
multiple corpora and exposes English top-word lists plus Zipf frequencies, which
are suitable for stable IME cost assignment.

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

### Manual layer

Manual TSV files are explicit additions and are not filtered by
`deny_words.tsv`.

- `data/manual/basic_words.tsv`: high-priority function words, pronouns,
  auxiliaries, prepositions, and everyday adverbs.
- `data/manual/contractions.tsv`: apostrophe outputs such as `im -> I'm` and
  `dont -> don't`.
- `data/manual/spelling_variants.tsv`: common US, Canadian, and British spelling
  variants.
- `data/manual/tech_common_words.tsv`: common technology and product terms such
  as `app`, `email`, `Android`, `GitHub`, `OpenAI`, and `AutoCAD`.
- `data/manual/deny_words.tsv`: project-controlled exclusions for wordfreq,
  corpus, and optional SCOWL layers. Lines beginning with `#` are comments.

### Optional SCOWL / English Speller Database

SCOWL is optional. To use it, place a plain one-word-per-line word list at:

```text
external_sources/scowl/words.txt
```

If the file is absent, generation continues with a warning. SCOWL is used as an
optional spell validation, optional enrichment, and coverage-report source.
wordfreq words are not automatically removed just because they are absent from
SCOWL.

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

The corpus layer no longer filters `token.is_stop`; it only requires
alphabetic tokens and then applies the same normalization and deny-word rules as
the wordfreq layer.

## Scoring and merge rules

- Manual scores are authored directly in the TSV files.
- wordfreq scores are derived from Zipf frequency and rank, with common words
  receiving lower costs.
- Corpus scores keep the existing `-log(count / total)` idea and map corpus
  words into a lower-priority integer range so rare corpus-only words do not
  outrank manual or wordfreq entries.
- Scores are clamped to `0..32767`.
- Merge key is `input_word + "\t" + output_word`.
- Duplicate keys keep the lower score.
- The same `input_word` may produce multiple `output_word` candidates, so
  `were -> were` and `were -> we're` both remain.

## Build

Install dependencies:

```bash
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

The spaCy model is only required when `ENABLE_CORPUS=1`.

Generate the dictionary:

```bash
python main.py
```

Verify the generated TSV or ZIP:

```bash
python verify_dictionary_output.py
```

## Outputs

- `1-grams_score_cost_pos_combined_with_ner.txt`: final dictionary TSV.
- `1-grams_score_cost_pos_combined_with_ner.zip`: ZIP package containing the
  TSV plus summary and notice/license files when present.
- `dictionary_source_summary.json`: source counts, configuration, wordfreq
  coverage, SCOWL status, corpus status, and merge counts.

## Licenses and notices

Project code is licensed under Apache License 2.0. See `LICENSE`.

Generated dictionary data includes wordfreq-derived data. The upstream
`wordfreq` repository states that the code is redistributable under Apache
License 2.0 and that it includes data files redistributable under Creative
Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0), with additional
source notices documented upstream.

Relevant local notice files:

- `NOTICE.md`
- `LICENSES/wordfreq-NOTICE.md`
- `LICENSES/CC-BY-SA-4.0.txt`

When distributing generated dictionary data, include the generated ZIP together
with these notices and licenses.
