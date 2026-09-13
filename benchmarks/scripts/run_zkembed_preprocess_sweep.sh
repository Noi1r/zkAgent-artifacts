#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

if [[ $# -ne 0 ]]; then
    echo "Usage: THREADS=32 bash benchmarks/scripts/run_zkembed_preprocess_sweep.sh"
    if [[ $# -eq 1 && ( "$1" == "--help" || "$1" == "-h" ) ]]; then exit 0; fi
    exit 2
fi

THREADS="${THREADS:-${OMP_NUM_THREADS:-32}}"
if [[ ! "${THREADS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "THREADS must be a positive integer." >&2
    exit 2
fi
export THREADS
export OMP_NUM_THREADS="${THREADS}"
export OMP_MAX_ACTIVE_LEVELS=1

BUILD_DIR="${BUILD_DIR:-${ROOT}/cmake-build-release}"
[[ "${BUILD_DIR}" = /* ]] || BUILD_DIR="${ROOT}/${BUILD_DIR}"
BIN="${BUILD_DIR}/src/benchmark_zkembed_preprocess"
if [[ ! -x "${BIN}" ]]; then
    echo "Missing benchmark binary: ${BIN}" >&2
    exit 1
fi

cd "${ROOT}"
MODELS=(tinyllama gpt2-medium gpt2-xl gpt3 deepseek-v2 deepseek-v3)
OUT_DIR="${ROOT}/benchmark_logs/table3/$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "${ROOT}/benchmark_logs/table3"
mkdir "${OUT_DIR}"
SUMMARY="${OUT_DIR}/summary.tsv"
printf 'model\tstatus\texit_code\n' > "${SUMMARY}"
FAILED=0

trap 'exit 130' INT
trap 'exit 143' TERM

for MODEL in "${MODELS[@]}"; do
    LOG="${OUT_DIR}/${MODEL}.log"
    RESOURCE="${OUT_DIR}/${MODEL}.resources.json"
    echo "Running Table 3: ${MODEL}, threads=${THREADS}"
    if ZKEMBED_MODEL="${MODEL}" BENCHMARK_REPORT_PATH="${OUT_DIR}/${MODEL}.md" \
        python3 "${ROOT}/scripts/measure_resources.py" \
        --label "table3-${MODEL}" --output "${RESOURCE}" -- "${BIN}" \
        2>&1 | tee "${LOG}"; then
        EXIT_CODE=0
    else
        CODES=("${PIPESTATUS[@]}")
        EXIT_CODE="${CODES[0]}"
        if [[ "${CODES[1]}" -ne 0 ]]; then
            echo "Could not finish writing ${LOG}; stopping." >&2
            exit "${CODES[1]}"
        fi
    fi

    if [[ "${EXIT_CODE}" -eq 0 ]]; then
        STATUS=COMPLETED
    else
        STATUS=FAILED
        FAILED=1
    fi
    printf '%s\t%s\t%s\n' "${MODEL}" "${STATUS}" "${EXIT_CODE}" >> "${SUMMARY}"
    echo "${MODEL}: ${STATUS} (exit ${EXIT_CODE})"
    if [[ "${EXIT_CODE}" -eq 130 || "${EXIT_CODE}" -eq 143 ]]; then
        exit "${EXIT_CODE}"
    fi
done

echo "Results: ${OUT_DIR}"
cat "${SUMMARY}"
exit "${FAILED}"
