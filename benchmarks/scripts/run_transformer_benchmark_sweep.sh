#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
ROOT="$(pwd)"

BUILD_DIR="${ROOT}/cmake-build-release"
BIN="${BUILD_DIR}/src/benchmark_transformer"
MODEL_DIR="${ROOT}/model/gpt2-124m"
TOKEN_DIR="${ROOT}/demo/table5"
CACHE_DIR="${ROOT}/cache"
THREADS=32
SEQ_LENS="32 64 128 256 512"
LOOKUP_BACKENDS="shout logup-gkr"
PROOF_PROFILE="table5-zkgpt-standalone"
REPORT_READER="${ROOT}/benchmarks/scripts/benchmark_report.py"
MEASURE="${ROOT}/scripts/measure_resources.py"

if [[ $# -ne 0 ]]; then
    echo "Table 5 takes no arguments; run bash benchmarks/run.sh table5." >&2
    exit 2
fi

if [[ ! -x "${BIN}" ]]; then
    echo "benchmark binary not found: ${BIN}" >&2
    exit 1
fi

for T in ${SEQ_LENS}; do
    TOKEN_FILE="${TOKEN_DIR}/gpt2_tokens_T${T}.txt"
    if [[ ! -f "${TOKEN_FILE}" ]]; then
        echo "missing bundled Table 5 token file: ${TOKEN_FILE}" >&2
        exit 1
    fi
done
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
OUT_DIR="${ROOT}/benchmark_logs/table5/${STAMP}"
mkdir -p "${ROOT}/benchmark_logs/table5" "${CACHE_DIR}"
mkdir "${OUT_DIR}"
TSV="${OUT_DIR}/${PROOF_PROFILE}.tsv"
python3 "${REPORT_READER}" --header > "${TSV}"

for BACKEND in ${LOOKUP_BACKENDS}; do
  for T in ${SEQ_LENS}; do
    TOKEN_FILE="${TOKEN_DIR}/gpt2_tokens_T${T}.txt"
    if [[ ! -f "${TOKEN_FILE}" ]]; then
        echo "missing token file: ${TOKEN_FILE}" >&2
        exit 1
    fi
    LOG="${OUT_DIR}/${BACKEND}_T${T}.log"
    args=(
        --model-dir "${MODEL_DIR}"
        --token-file "${TOKEN_FILE}"
        --seq-len "${T}"
        --threads "${THREADS}"
        --cache-dir "${CACHE_DIR}"
        --lookup-backend "${BACKEND}"
        --proof-profile "${PROOF_PROFILE}"
        --use-cache
    )

    echo "Running ${PROOF_PROFILE}: ${BACKEND}, T=${T}, threads=${THREADS}"
    python3 "${MEASURE}" --label "table5-${BACKEND}-T${T}" \
        --output "${LOG}.resources.json" -- "${BIN}" "${args[@]}" 2>&1 | tee "${LOG}"
    if [[ "${BACKEND}" == "shout" ]]; then LOOKUP_LABEL=SHOUT; else LOOKUP_LABEL=LogUp-GKR; fi
    python3 "${REPORT_READER}" "${LOG}" \
      --resources "${LOG}.resources.json" \
      --expect "profile=${PROOF_PROFILE}" --expect "sequence=${T}" \
      --expect "threads=${THREADS}" \
      --expect "lookup=${LOOKUP_LABEL}, max-chunks=2" \
      --expect "lookup_openings=batch-mvkzg" \
      --expect "transcript=Fiat-Shamir" >> "${TSV}"
  done
done

echo "results: ${TSV}"
