#!/bin/bash
# Network pipeline (about 35 min on one core).  Run after scripts/make_figure1.py (or use ../run_all.sh).
set -e
cd "$(dirname "$0")/.."
python scripts/run_ca1_analyses.py
python scripts/run_ablation.py
python scripts/eta_map.py
python scripts/make_figure2.py && python scripts/make_figure3.py && python scripts/make_figure4.py && python scripts/make_figureS1.py && python scripts/eta_I_sensitivity.py && python scripts/make_figureS2.py
