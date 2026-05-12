#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"
COMMON_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

FILTER_FILE="${FILTER_FILE:-${PROJECT_DIR}/instances_verified500.txt}"
if [[ ! -f "${FILTER_FILE}" ]]; then
  echo "Error: filter file not found: ${FILTER_FILE}" >&2
  exit 1
fi

TRANSLATED_FILE="${TRANSLATED_FILE:-}"
if [[ -z "${TRANSLATED_FILE}" ]]; then
  if [[ -f "${PROJECT_DIR}/re_issues/llm_all.json" ]]; then
    TRANSLATED_FILE="${PROJECT_DIR}/re_issues/llm_all.json"
  elif [[ -f "${COMMON_ROOT}/re_issues/llm_all.json" ]]; then
    TRANSLATED_FILE="${COMMON_ROOT}/re_issues/llm_all.json"
  elif [[ -f "${PROJECT_DIR}/re_issues/llm_django.json" ]]; then
    TRANSLATED_FILE="${PROJECT_DIR}/re_issues/llm_django.json"
  elif [[ -f "${COMMON_ROOT}/re_issues/llm_django.json" ]]; then
    TRANSLATED_FILE="${COMMON_ROOT}/re_issues/llm_django.json"
  fi
fi

LEVEL2_MAPPING_DIR="${LEVEL2_MAPPING_DIR:-${PROJECT_DIR}/output/level2_repo_maps_verified500}"
if [[ ! -d "${LEVEL2_MAPPING_DIR}" ]]; then
  echo "Error: Level 1/2 mapping dir not found: ${LEVEL2_MAPPING_DIR}" >&2
  exit 1
fi

LEVEL3_VERIFIED_STORE="${LEVEL3_VERIFIED_STORE:-${PROJECT_DIR}/output/verified_perturbations/level3}"
if [[ ! -d "${LEVEL3_VERIFIED_STORE}" ]]; then
  echo "Error: Level 3 verified store not found: ${LEVEL3_VERIFIED_STORE}" >&2
  exit 1
fi

LEVEL3_TRAJ_DIR="${LEVEL3_TRAJ_DIR:-${PROJECT_DIR}/../v2_baseline/mini-swe-agent/trajectories_baseline_gpt54mini}"
if [[ ! -d "${LEVEL3_TRAJ_DIR}" ]]; then
  LEVEL3_TRAJ_DIR="/data/swebench/silinchen/Silin-SWE-Bench/v2_baseline/mini-swe-agent/trajectories_baseline_gpt54mini"
fi

LEVEL3_REPO_ROOT="${LEVEL3_REPO_ROOT:-/data/swebench/workspace_henglian/SWE-Search/tmp/repos}"
if [[ ! -d "${LEVEL3_REPO_ROOT}" ]]; then
  echo "Error: Level 3 repo root not found: ${LEVEL3_REPO_ROOT}" >&2
  exit 1
fi

LEVEL2_SEMANTIC_SEED="${LEVEL2_SEMANTIC_SEED:-42}"
LEVEL3_SEED="${LEVEL3_SEED:-42}"
MAPPED_MODEL_CONFIG="${MAPPED_MODEL_CONFIG:-swebench_qingyun_gpt54mini.yaml}"
MAPPED_MODEL_NAME="${MAPPED_MODEL_NAME:-}"
MAPPED_LITELLM_REGISTRY_PATH="${MAPPED_LITELLM_REGISTRY_PATH:-${PROJECT_DIR}/litellm_registry_gpt54mini.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/trajectories_level1_level2_level3}"
mkdir -p "${OUTPUT_DIR}"
RUN_LOG="${OUTPUT_DIR}/run_$(date +%Y%m%d_%H%M%S).log"

cd "${PROJECT_DIR}"
export PYTHONUNBUFFERED=1
export MSWEA_DATASETS_LOCAL_ONLY="${MSWEA_DATASETS_LOCAL_ONLY:-1}"
export TMPDIR="${PROJECT_DIR}/tmp"
mkdir -p "${TMPDIR}"

echo "Full run log: ${RUN_LOG}" >&2
if [[ "${MAPPED_RUN_TEE:-0}" != "1" ]]; then
  echo "Benchmark stdout/stderr -> log only. Watch: tail -f \"${RUN_LOG}\"" >&2
fi
echo "Run mode: nohup background (set MAPPED_RUN_BACKGROUND=0 for foreground)." >&2

_run() {
  local -a model_args=()
  local -a translated_args=()
  local -a level3_traj_args=()

  if [[ -n "${MAPPED_MODEL_NAME}" ]]; then
    model_args+=(--model "${MAPPED_MODEL_NAME}")
  fi
  if [[ -n "${TRANSLATED_FILE}" && -f "${TRANSLATED_FILE}" ]]; then
    translated_args=(--translated-problems "${TRANSLATED_FILE}")
  fi
  if [[ -d "${LEVEL3_TRAJ_DIR}" ]]; then
    level3_traj_args=(--level3-trajectories "${LEVEL3_TRAJ_DIR}")
  fi

  echo "=== Level 1+2+3 started $(date -Iseconds) ==="
  echo "filter-file: ${FILTER_FILE}"
  if [[ ${#translated_args[@]} -gt 0 ]]; then
    echo "translated-problems: ${TRANSLATED_FILE}"
  else
    echo "translated-problems: <not provided; using original problem statements>"
  fi
  echo "level2-mapping-dir: ${LEVEL2_MAPPING_DIR}"
  echo "level2-semantic-seed: ${LEVEL2_SEMANTIC_SEED}"
  echo "level3-verified-store: ${LEVEL3_VERIFIED_STORE}"
  if [[ -d "${LEVEL3_TRAJ_DIR}" ]]; then
    echo "level3-trajectories: ${LEVEL3_TRAJ_DIR}"
  else
    echo "level3-trajectories: <not provided; using verified store only>"
  fi
  echo "level3-repo-root: ${LEVEL3_REPO_ROOT}"
  echo "level3-seed: ${LEVEL3_SEED}"
  echo "model-config: ${MAPPED_MODEL_CONFIG}"
  echo "litellm-registry: ${MAPPED_LITELLM_REGISTRY_PATH}"
  echo "output: ${OUTPUT_DIR}"
  LITELLM_MODEL_REGISTRY_PATH="${MAPPED_LITELLM_REGISTRY_PATH}" \
  PYTHONPATH="src:.:${PYTHONPATH:-}" python3 -m minisweagent.run.benchmarks.swebench_mapped \
    -c swebench.yaml \
    -c "${MAPPED_MODEL_CONFIG}" \
    --subset verified \
    --split test \
    --workers 24 \
    --filter-file "${FILTER_FILE}" \
    --semantic-mappings "${LEVEL2_MAPPING_DIR}" \
    --semantic-seed "${LEVEL2_SEMANTIC_SEED}" \
    --enabled-layer identity_l1 \
    --enabled-layer namespace_l2 \
    --level3-mode intra_file_reorder \
    --level3-repo-root "${LEVEL3_REPO_ROOT}" \
    --level3-verified-store "${LEVEL3_VERIFIED_STORE}" \
    --level3-seed "${LEVEL3_SEED}" \
    --level3-submission-mode keep \
    "${translated_args[@]}" \
    "${level3_traj_args[@]}" \
    "${model_args[@]}" \
    --output "${OUTPUT_DIR}"
  echo "=== Level 1+2+3 finished $(date -Iseconds) ==="
}

export FILTER_FILE TRANSLATED_FILE LEVEL2_MAPPING_DIR LEVEL2_SEMANTIC_SEED LEVEL3_VERIFIED_STORE LEVEL3_TRAJ_DIR LEVEL3_REPO_ROOT LEVEL3_SEED MAPPED_MODEL_CONFIG MAPPED_MODEL_NAME MAPPED_LITELLM_REGISTRY_PATH OUTPUT_DIR PROJECT_DIR TMPDIR

if [[ "${MAPPED_RUN_BACKGROUND:-1}" == "1" ]]; then
  export -f _run
  nohup bash -c '_run' >> "${RUN_LOG}" 2>&1 &
  echo "$!" > "${OUTPUT_DIR}/latest_run.pid"
  echo "Started in background PID $!, log: ${RUN_LOG}" >&2
  echo "PID file: ${OUTPUT_DIR}/latest_run.pid" >&2
  echo "Watch: tail -f \"${RUN_LOG}\"" >&2
  exit 0
fi

if [[ "${MAPPED_RUN_TEE:-0}" == "1" ]]; then
  _run 2>&1 | tee -a "${RUN_LOG}"
else
  _run >> "${RUN_LOG}" 2>&1
fi
