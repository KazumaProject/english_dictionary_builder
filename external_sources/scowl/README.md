# Optional SCOWL / English Speller Database source

Place a plain one-word-per-line SCOWL / English Speller Database word list at:

```text
external_sources/scowl/words.txt
```

`main.py` reads this file only when it exists. If the file is missing, dictionary
generation continues with a warning. This repository intentionally does not
create or invent `words.txt`.

SCOWL is treated as an optional spell validation, optional enrichment, and
coverage-report source. Words that are absent from SCOWL are not automatically
removed from the wordfreq layer.

When adding SCOWL locally, also place the applicable upstream license or
copyright notice in:

```text
external_sources/scowl/LICENSE_OR_COPYRIGHT.txt
```

Project page: http://wordlist.aspell.net/

