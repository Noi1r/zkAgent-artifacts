#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Usage: bash build.sh [Debug|Release|RelWithDebInfo|MinSizeRel] [CMake options...]"
    echo "Environment: BUILD_DIR (absolute or project-relative), JOBS (default: 4)."
    exit 0
fi
BUILD_TYPE="${1:-Debug}"
if [[ $# -gt 0 ]]; then shift; fi
case "${BUILD_TYPE}" in
    Debug|Release|RelWithDebInfo|MinSizeRel) ;;
    *) echo "Unsupported build type: ${BUILD_TYPE}" >&2; exit 2 ;;
esac
BUILD_DIR="${BUILD_DIR:-cmake-build-$(printf '%s' "${BUILD_TYPE}" | tr '[:upper:]' '[:lower:]')}"
[[ "${BUILD_DIR}" = /* ]] || BUILD_DIR="${ROOT}/${BUILD_DIR}"

cmake -S "${ROOT}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" "$@"
cmake --build "${BUILD_DIR}" --parallel "${JOBS:-4}"
