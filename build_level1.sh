#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"

FILTER_FILE="${FILTER_FILE:-${PROJECT_DIR}/instances_verified500.txt}"
if [[ ! -f "${FILTER_FILE}" ]]; then
  echo "Error: filter file not found: ${FILTER_FILE}" >&2
  exit 1
fi

OUTPUT_FILE="${OUTPUT_FILE:-${PROJECT_DIR}/re_issues/llm_verified500.json}"
L1_MODEL_NAME="${L1_MODEL_NAME:-gpt-5.4-mini}"
L1_WORKERS="${L1_WORKERS:-8}"
L1_RESUME="${L1_RESUME:-1}"

cd "${PROJECT_DIR}"
export PYTHONUNBUFFERED=1

CMD=(
  python3 glasses/translate_verified_problems.py
  --filter-file "${FILTER_FILE}"
  --output "${OUTPUT_FILE}"
  --model "${L1_MODEL_NAME}"
  --workers "${L1_WORKERS}"
)

if [[ "${L1_RESUME}" == "1" ]]; then
  CMD+=(--resume)
fi

PYTHONPATH="src:.:glasses:${PYTHONPATH:-}" "${CMD[@]}"
