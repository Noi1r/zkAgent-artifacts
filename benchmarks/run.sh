#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MEASURE="${ROOT}/scripts/measure_resources.py"
usage() {
    cat <<'EOF'
Usage: bash benchmarks/run.sh <table1|table2|table3|table4|table5|table6|decode> [arguments]

  table1  GPT-2 zkEmbed preprocessing (ours).
  table2  zkEmbed online benchmark.
  table3  Cross-model preprocessing; set ZKEMBED_MODEL to one model or all.
  table4  Thaler / Hadamard RoPE core.
          Arguments: [seq_len head_dim [heads]].
          No arguments: all T=32,128,512 and d=64,128 combinations, H=12,
          then Llama-2-7B RoPE scale (T=512, d=128, H=32).
          Always runs both methods with one thread.
  table5  Transformer component benchmark; reads bundled tokens from demo/table5.
  table6  Greedy inference core; existing benchmark_inference arguments still work.
          --workload weather|coding selects a bundled Table 6 configuration.
          Add --with-tools to run Coral and the workload's zkTLS/SP1 component.
          Add --skip-inference with --with-tools to run only the tool components.
          No arguments prints benchmark_inference help.
          Pipeline help: python3 benchmarks/scripts/run_table6.py --help
  decode  Fixed zkDecode benchmark: N=50257, T=512, k=40, both backends.

Table 5 and decode use cmake-build-release. Other entries accept BUILD_DIR
(absolute or project-relative; default: cmake-build-release).
Build and prepare the model first. Table 5 uses the fixed token files in demo/table5.
See README.md for setup, data provenance, output, and timing scopes.
Each invocation saves logs, reports, resource JSON and any summary together in
benchmark_logs/<table1..table6|decode>/<timestamp>_<pid>/.
EOF
}
if [[ $# -eq 0 || "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi
CASE="$1"
shift
case "${CASE}" in
    table1|table2|table3|table4|table5|table6|decode) ;;
    *) usage >&2; echo "Unknown benchmark: ${CASE}" >&2; exit 2 ;;
esac
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi
BUILD_DIR="${BUILD_DIR:-${ROOT}/cmake-build-release}"
[[ "${BUILD_DIR}" = /* ]] || BUILD_DIR="${ROOT}/${BUILD_DIR}"
export BUILD_DIR
cd "${ROOT}"
require_binary() {
    if [[ ! -x "${BUILD_DIR}/src/$1" ]]; then
        echo "Missing binary: ${BUILD_DIR}/src/$1" >&2
        echo "Build first: BUILD_DIR='${BUILD_DIR}' bash build.sh Release -DBUILD_BENCHMARKS=ON" >&2
        exit 1
    fi
}
run_measured() {
    local label="$1"
    shift
    local out_dir="${ROOT}/benchmark_logs/${CASE}/$(date +%Y%m%d_%H%M%S)_$$"
    mkdir -p "${ROOT}/benchmark_logs/${CASE}"
    mkdir "${out_dir}"
    echo "Results: ${out_dir}"
    BENCHMARK_REPORT_PATH="${out_dir}/report.md" \
        python3 "${MEASURE}" --label "${label}" \
        --output "${out_dir}/resources.json" -- "$@" \
        2>&1 | tee "${out_dir}/run.log"
}
case "${CASE}" in
    table1|table3)
        if [[ $# -ne 0 ]]; then echo "${CASE} uses environment variables, not positional arguments." >&2; exit 2; fi
        if [[ "${CASE}" == "table1" ]]; then
            export ZKEMBED_MODEL=gpt2-small
        elif [[ "${ZKEMBED_MODEL:-}" == "all" ]]; then
            exec "${BASH}" "${ROOT}/benchmarks/scripts/run_zkembed_preprocess_sweep.sh"
        elif [[ -z "${ZKEMBED_MODEL:-}" ]]; then
            echo "Set ZKEMBED_MODEL to one model (for example tinyllama) or all." >&2
            exit 2
        fi
        require_binary benchmark_zkembed_preprocess
        run_measured "${CASE}-${ZKEMBED_MODEL}" \
            "${BUILD_DIR}/src/benchmark_zkembed_preprocess"
        ;;
    table2)
        require_binary benchmark_zkembed
        run_measured table2 "${BUILD_DIR}/src/benchmark_zkembed" "$@"
        ;;
    table4)
        require_binary benchmark_zkposenc
        run_measured table4 "${BUILD_DIR}/src/benchmark_zkposenc" "$@"
        ;;
    table5)
        if [[ $# -ne 0 ]]; then echo "table5 takes no arguments." >&2; exit 2; fi
        BUILD_DIR="${ROOT}/cmake-build-release"
        require_binary benchmark_transformer
        exec "${BASH}" "${ROOT}/benchmarks/scripts/run_transformer_benchmark_sweep.sh"
        ;;
    decode)
        if [[ $# -ne 0 ]]; then echo "decode takes no arguments." >&2; exit 2; fi
        exec python3 "${ROOT}/benchmarks/scripts/run_decode_benchmark.py"
        ;;
    table6)
        for arg in "$@"; do
            case "${arg}" in
                --workload|--workload=*|--with-tools|--skip-inference|--output-dir|--output-dir=*|--coral-ptau|--coral-ptau=*)
                    exec python3 "${ROOT}/benchmarks/scripts/run_table6.py" "$@"
                    ;;
            esac
        done
        BIN=benchmark_inference
        require_binary "${BIN}"
        if [[ $# -eq 0 ]]; then exec "${BUILD_DIR}/src/${BIN}" --help; fi
        run_measured table6 "${BUILD_DIR}/src/${BIN}" "$@"
        ;;
esac
