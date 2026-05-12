#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/output/level2_repo_maps_verified500}"
API_BASE="${OPENAI_BASE_URL:-https://api.qingyuntop.top/v1}"
LEVEL2_SEED="${LEVEL2_SEED:-42}"
LEVEL2_VARIANT_COUNT="${LEVEL2_VARIANT_COUNT:-1}"
LEVEL2_MAPPING_MODE="${LEVEL2_MAPPING_MODE:-identity_namespace_l2}"
FILTER_FILE="${FILTER_FILE:-${PROJECT_DIR}/instances_verified500.txt}"

if [[ ! -f "${FILTER_FILE}" ]]; then
  echo "Error: filter file not found: ${FILTER_FILE}" >&2
  exit 1
fi

cd "${PROJECT_DIR}"
CMD=(
  python3 glasses/batch_map.py
  --filter-file "${FILTER_FILE}"
  --seed "${LEVEL2_SEED}"
  --variant-count "${LEVEL2_VARIANT_COUNT}"
  --model openai/gpt-5-mini
  --api-base "${API_BASE}"
  --mapping-mode "${LEVEL2_MAPPING_MODE}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  CMD+=(--api-key "${OPENAI_API_KEY}")
fi

PYTHONPATH="src:.:${PYTHONPATH:-}" "${CMD[@]}"
