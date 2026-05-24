# Notices for generated English IME dictionary data

This project generates one-gram and phrase English IME dictionary artifacts
from local manual entries, wordfreq-derived frequency data, generated
inflections validated against wordfreq or optional SCOWL, an optional corpus
layer, and an optional SCOWL / English Speller Database validation layer.

The generated dictionary files include data derived from `wordfreq`:

- Project: https://github.com/rspeer/wordfreq
- Package: https://pypi.org/project/wordfreq/
- Author / attribution name: Robyn Speer
- Code license: Apache License 2.0
- Data notice: wordfreq includes data files that may be redistributed under
  Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0),
  plus source-specific attribution notes documented upstream.

Changes made to source data in this project:

- filtering
- normalization
- conversion to one-gram dictionary entries
- conversion to separate manual phrase dictionary entries
- limited generation of explicitly configured inflection forms
- cost assignment
- merge with manual entries
- packaging as ZIP

Manual TSV files in this repository are project-authored data. Phrase/bigram
artifacts are generated from local manual phrase TSV files only in the current
build path.

The generated dictionary ZIPs should be distributed with this NOTICE file and
the files under `LICENSES/`.
