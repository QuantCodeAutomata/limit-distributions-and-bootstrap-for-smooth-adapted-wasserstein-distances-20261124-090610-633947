#!/usr/bin/env bash
# Runs the test suite and all three experiments at their reduced (default)
# configuration. See README.md for the paper-scale `--full` commands.
set -uo pipefail

python -m pytest -m "not market_data"

python -m exp.exp1
python -m exp.exp2
python -m exp.exp3
