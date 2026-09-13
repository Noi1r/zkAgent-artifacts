#!/usr/bin/env python3
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
BINARY = ROOT / "cmake-build-release/src/demo_zkdecode_run"
MEASURE = ROOT / "scripts/measure_resources.py"
VOCAB, TOKENS, TOPK, THREADS = 50257, 512, 40, 32
FIELDS = ("mode", "backend", "prover_s_per_token", "verifier_ms_per_token",
          "proof_kb_per_token", "status", "wall_s", "peak_rss_gib")


def read_result(log, mode, backend):
    records = [line for line in log.read_text().splitlines()
               if line.startswith("DECODE_RESULT ")]
    if len(records) != 1:
        raise ValueError(f"Missing or duplicate result in {log}")
    result = dict(item.split("=", 1) for item in records[0].split()[1:])
    expected = dict(mode=mode, backend=backend, vocab=str(VOCAB),
                    tokens=str(TOKENS), k=str(TOPK), threads=str(THREADS), verified="yes")
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Configuration or verification mismatch in {log}")
    prover = float(result["prover_seconds"])
    verifier = float(result["verifier_seconds"])
    proof = int(result["proof_bytes"])
    if not all(math.isfinite(x) and x >= 0 for x in (prover, verifier)) or proof < 0:
        raise ValueError(f"Invalid measurement in {log}")
    return dict(mode=mode, backend=backend, prover_s_per_token=prover / TOKENS,
                verifier_ms_per_token=verifier * 1000 / TOKENS,
                proof_kb_per_token=proof / 1024 / TOKENS, status="PASS")


def print_results(results):
    if not results:
        return
    headers = ("Mode", "Backend", "Prove (s/token)", "Verify (ms/token)",
               "Proof (KB/token)", "Wall (s)", "Peak RSS (GiB)", "Status")
    rows = [(
        row["mode"], row["backend"], f"{row['prover_s_per_token']:.6f}",
        f"{row['verifier_ms_per_token']:.6f}", f"{row['proof_kb_per_token']:.3f}",
        f"{row['wall_s']:.3f}", f"{row['peak_rss_gib']:.3f}", row["status"]
    ) for row in results]
    widths = [max(len(values[column]) for values in [headers, *rows])
              for column in range(len(headers))]
    print(f"\nResults ({len(rows)}/4 completed)")
    print(" | ".join(value.ljust(width) for value, width in zip(headers, widths)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(value.rjust(width) if 2 <= column <= 6 else value.ljust(width)
                         for column, (value, width) in enumerate(zip(row, widths))))


def main():
    if len(sys.argv) != 1:
        raise ValueError("Run bash benchmarks/run.sh decode without arguments")
    if not BINARY.is_file():
        raise FileNotFoundError(f"Build first: {BINARY}")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_{os.getpid()}"
    out = ROOT / "benchmark_logs" / "decode" / stamp
    out.mkdir(parents=True, exist_ok=False)
    print(f"zkDecode: synthetic logits, N={VOCAB}, T={TOKENS}, k={TOPK}, threads={THREADS}", flush=True)
    results = []
    try:
        with (out / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            for mode in ("greedy", "topk"):
                for backend, backend_option in (("SHOUT", "shout"), ("LogUp-GKR", "logup-gkr")):
                    log = out / f"{mode}_{backend}.log"
                    print(f"[{len(results) + 1}/4] Running {mode} / {backend}", flush=True)
                    command = [str(BINARY), "--decode", mode, "--vocab-size", str(VOCAB),
                               "--tokens", str(TOKENS), "--k", str(TOPK), "--threads", str(THREADS),
                               "--lookup-backend", backend_option, "--decode-seed", "20260421",
                               "--bound-bits", "20", "--lookup-g1-encoding", "compressed"]
                    try:
                        with log.open("w") as output:
                            subprocess.run([sys.executable, str(MEASURE), "--label",
                                            f"decode-{mode}-{backend}", "--output",
                                            str(log) + ".resources.json", "--", *command],
                                           cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, check=True)
                    except subprocess.CalledProcessError as error:
                        raise ValueError(f"{mode} / {backend} exited with code "
                                         f"{error.returncode}; see {log.name}") from error
                    row = read_result(log, mode, backend)
                    resources = json.loads(Path(str(log) + ".resources.json").read_text())
                    row.update(wall_s=resources["wall_seconds"], peak_rss_gib=resources["peak_rss_gib"])
                    writer.writerow(row)
                    handle.flush()
                    results.append(row)
    finally:
        print_results(results)
        print(f"\nLogs and summary.csv: {out.relative_to(ROOT)}/", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        sys.exit(f"Decoding benchmark failed: {error}")
