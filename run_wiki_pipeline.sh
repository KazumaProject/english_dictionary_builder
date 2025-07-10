#!/usr/bin/env bash
# ------------------------------------------------------------
# Download latest English Wikipedia dump
# → clean it with WikiExtractor (JSON lines, no tags/templates)
# → run main.py on the extracted text
#
# Usage:
#   chmod +x run_wiki_pipeline.sh
#   ./run_wiki_pipeline.sh
# ------------------------------------------------------------
set -euo pipefail

# ─── Settings ────────────────────────────────────────────────
DUMP_URL="https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2"
DUMP_FILE="enwiki-latest-pages-articles.xml.bz2"
EXTRACT_DIR="wiki_json"              # main.py 側と合わせる
PYTHON="python3"                     # 必要に応じて変更
# ------------------------------------------------------------

echo ">>> 1) Installing Python deps"
$PYTHON -m pip install --quiet --upgrade pip
$PYTHON -m pip install --quiet wikiextractor spacy tqdm numpy datasets
$PYTHON -m spacy download --quiet en_core_web_sm

echo ">>> 2) Downloading Wikipedia dump"
[ -f "$DUMP_FILE" ] || wget -c "$DUMP_URL"

echo ">>> 3) Running WikiExtractor  (this may take a while…)"
wikiextractor --json --processes "$(nproc)" -b 250M \
  -o "$EXTRACT_DIR" --no-templates "$DUMP_FILE"

echo ">>> 4) Launching main.py"
$PYTHON main.py

echo "✅ Pipeline finished"
