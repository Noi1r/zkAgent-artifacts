#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

BUILD_DIR="${BUILD_DIR:-cmake-build-release}"
BIN="${ROOT}/${BUILD_DIR}/src/warmup_srs"

MV_MAX="${MV_MAX:-19}"
UNI_MAX_LOG_N="${UNI_MAX_LOG_N:-16}"
CACHE_DIR="${CACHE_DIR:-${ROOT}/cache}"
THREADS="${THREADS:-32}"

if [[ ! -x "${BIN}" ]]; then
    echo "[build] ${BIN} not found, invoking build.sh Release..."
    bash "${ROOT}/build.sh" Release
fi

mkdir -p "${CACHE_DIR}"
echo "Warming SRS into ${CACHE_DIR}"
echo "  mv_max=${MV_MAX} uni_max_log_n=${UNI_MAX_LOG_N} threads=${THREADS}"

OMP_NUM_THREADS="${THREADS}" "${BIN}" "${MV_MAX}" "${UNI_MAX_LOG_N}" "${CACHE_DIR}"

echo
echo "Cache contents:"
ls -lh "${CACHE_DIR}" | grep -E "mvkzg_m|kzg_N" || true
echo
echo "Total cache size:"
du -sh "${CACHE_DIR}"
