#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/debug.yaml}"
RUN_DIR="${2:-runs/debug}"

mkdir -p "${RUN_DIR}"
wdh seed-bank --config "${CONFIG}" --output "${RUN_DIR}/seed_bank.jsonl"
wdh profile \
  --config "${CONFIG}" \
  --candidates "${RUN_DIR}/seed_bank.jsonl" \
  --output "${RUN_DIR}/profiles.jsonl"
wdh select \
  --profiles "${RUN_DIR}/profiles.jsonl" \
  --output "${RUN_DIR}/team_sc.json" \
  --method sc_greedy \
  --team-size 4
wdh evaluate \
  --config "${CONFIG}" \
  --team "${RUN_DIR}/team_sc.json" \
  --output-dir "${RUN_DIR}/evaluation"
