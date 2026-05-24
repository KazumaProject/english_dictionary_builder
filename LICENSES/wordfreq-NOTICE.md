# wordfreq notice

This project uses `wordfreq` as the main vocabulary and frequency-ranking
source for generated English IME dictionary data.

- Upstream repository: https://github.com/rspeer/wordfreq
- Upstream README: https://github.com/rspeer/wordfreq/blob/master/README.md
- Upstream license file: https://github.com/rspeer/wordfreq/blob/master/LICENSE.txt
- Upstream notice file: https://github.com/rspeer/wordfreq/blob/master/NOTICE.md
- PyPI package: https://pypi.org/project/wordfreq/
- Citation requested by upstream: Robyn Speer. (2022). rspeer/wordfreq: v3.0
  (v3.0.2). Zenodo. https://doi.org/10.5281/zenodo.7199437

The upstream `LICENSE.txt` states that the Apache License 2.0 applies to the
code only and points to `NOTICE.md` for details about code and data licensing.
The upstream README and NOTICE state that `wordfreq` is freely redistributable
under the Apache License and includes data files that may be redistributed under
Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0).

The generated dictionary data in this repository contains wordfreq-derived
entries. This project transforms wordfreq-derived data by filtering,
normalizing, converting words to one-gram dictionary entries, assigning costs,
merging with manual entries, and packaging the result as a ZIP file.

