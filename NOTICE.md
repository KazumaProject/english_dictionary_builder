# Notices for generated English IME dictionary data

This project generates a one-gram English IME dictionary from local manual
entries, wordfreq-derived frequency data, an optional corpus layer, and an
optional SCOWL / English Speller Database validation layer.

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
- cost assignment
- merge with manual entries
- packaging as ZIP

The generated dictionary ZIP should be distributed with this NOTICE file and
the files under `LICENSES/`.

