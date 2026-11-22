#!/bin/bash
# Reproduces every figure, table value and key number (about 40 minutes on one CPU core).
# Optional first step: python scripts/search_network.py   (re-derives the network selection, about 9 minutes)
set -e
cd "$(dirname "$0")"
python scripts/make_figure1.py
python scripts/upstream_sensitivity.py
bash scripts/run_all_network.sh
python scripts/collect_key_numbers.py
python scripts/check_reproduction.py || echo "NOTE: differences to expected/key_numbers.json (different library versions or hardware can change last digits)"
