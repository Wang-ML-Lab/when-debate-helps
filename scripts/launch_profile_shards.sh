#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 4 ]]; then
  echo "usage: $0 CONFIG SEED_BANK OUTPUT_DIR GPU [GPU ...]" >&2
  exit 2
fi

CONFIG="$1"
SEED_BANK="$2"
OUTPUT_DIR="$3"
shift 3
GPUS=("$@")
NUM_SHARDS="${#GPUS[@]}"

mkdir -p "${OUTPUT_DIR}"
PIDS=()
for SHARD_ID in "${!GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES="${GPUS[$SHARD_ID]}" wdh profile \
    --config "${CONFIG}" \
    --set model.device=cuda:0 \
    --candidates "${SEED_BANK}" \
    --output "${OUTPUT_DIR}/profile-${SHARD_ID}.jsonl" \
    --shard-id "${SHARD_ID}" \
    --num-shards "${NUM_SHARDS}" \
    >"${OUTPUT_DIR}/profile-${SHARD_ID}.log" 2>&1 &
  PIDS+=("$!")
done

FAILED=0
for SHARD_ID in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$SHARD_ID]}"; then
    echo "profile shard ${SHARD_ID} failed; see ${OUTPUT_DIR}/profile-${SHARD_ID}.log" >&2
    FAILED=1
  fi
done
if [[ "${FAILED}" -ne 0 ]]; then
  exit 1
fi

INPUTS=()
for SHARD_ID in "${!GPUS[@]}"; do
  INPUTS+=("${OUTPUT_DIR}/profile-${SHARD_ID}.jsonl")
done
wdh merge-profiles \
  --inputs "${INPUTS[@]}" \
  --output "${OUTPUT_DIR}/profiles.jsonl" \
  --candidates "${SEED_BANK}"
