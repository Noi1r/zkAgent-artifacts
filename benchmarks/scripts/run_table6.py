#!/usr/bin/env python3

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

from table6_inputs import check_tool_inputs


ROOT = Path(__file__).resolve().parents[2]
CORAL = ROOT / "tools/coral"
PY2ZKVM = ROOT / "tools/py2zkvm"
ZKTLS = ROOT / "tools/zktls/weather-zktls-example"
WORKLOADS = {
    "weather": {"length": 152, "decode_batch": 32, "tool": "zktls", "document": "weather.json",
                "grammar": "weather.pest", "city": "Shanghai", "latitude": 31.2222, "longitude": 121.4581},
    "coding": {"length": 143, "decode_batch": 0, "tool": "sp1", "document": "code.json", "grammar": "coding.pest"},
}


def absolute(value):
    path = Path(value).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the bundled Table 6 inference and optional tool benchmarks.", allow_abbrev=False)
    parser.add_argument("--workload", type=str.lower, choices=WORKLOADS, required=True)
    parser.add_argument("--with-tools", action="store_true", help="also run Coral and zkTLS (weather) or SP1 (coding)")
    parser.add_argument("--skip-inference", action="store_true", help="with --with-tools, run only the tool components")
    parser.add_argument("--model-dir", type=absolute, default=ROOT / "model/gpt2-124m")
    parser.add_argument("--threads", type=positive_int, default=32)
    parser.add_argument("--lookup-backend", choices=("shout", "logup-gkr"), default="shout")
    parser.add_argument("--cache-dir", type=absolute, default=ROOT / "cache")
    parser.add_argument("--output-dir", type=absolute, help="new directory for logs, proofs and summary.json/summary.md")
    parser.add_argument("--coral-ptau", type=absolute, help="existing PPoT file; defaults to CORAL_PTAU or tools/coral/ppot_0080_23.ptau")
    args, extra = parser.parse_known_args(argv)
    if args.skip_inference and not args.with_tools:
        parser.error("--skip-inference requires --with-tools")
    if args.skip_inference and extra:
        parser.error("--skip-inference does not accept extra inference arguments")
    reserved = {"--token-file", "--seq-len", "--prompt-len", "--generation-mask", "--proof-profile",
                "--prepare-circuit-only", "--witness-only", "--prepare-lookup-routing-only"}
    for item in extra:
        if item.split("=", 1)[0] in reserved:
            parser.error(f"{item} is incompatible with the bundled workload; use inference-only arguments without --workload")
    args.extra = extra
    return args


def preflight(args, paths):
    errors = []

    def check(condition, message):
        if not condition:
            errors.append(message)

    def executable(path, label):
        check(path.is_file() and os.access(path, os.X_OK), f"Missing {label}: {path}")

    if not args.skip_inference:
        executable(paths["inference"], "benchmark_inference; build the artifact first")
        check(args.model_dir.is_dir(), f"Missing prepared model: {args.model_dir}")
        for key in ("tokens", "mask"):
            check(paths[key].is_file(), f"Missing {key}: {paths[key]}")
    if not args.with_tools:
        return errors
    executable(paths["coral"], "Coral binary; follow tools/coral/README.md")
    check(paths["ptau"].is_file() and paths["ptau"].stat().st_size > 0,
          f"Missing Coral PPoT file: {paths['ptau']} (set --coral-ptau or CORAL_PTAU)")
    for key in ("document", "grammar"):
        check(paths[key].is_file(), f"Missing Coral {key}: {paths[key]}")
    if args.workload == "weather":
        check(shutil.which("node"), "Missing Node.js; follow tools/zktls/weather-zktls-example/README.md")
        if shutil.which("node"):
            script = ("const {createRequire} = await import('node:module'); "
                      "createRequire(import.meta.url).resolve('@primuslabs/zktls-core-sdk'); "
                      "const {configDotenv} = await import('dotenv'); configDotenv({quiet:true}); "
                      "process.exit(process.env.APP_ID && process.env.APP_SECRET ? 0 : 1);")
            try:
                result = subprocess.run(["node", "--input-type=module", "-e", script], cwd=ZKTLS,
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
                check(result.returncode == 0,
                      "zkTLS dependencies/configuration missing: run npm install and set APP_ID/APP_SECRET in its .env or environment")
            except subprocess.TimeoutExpired:
                check(False, "zkTLS dependency/configuration check timed out after 30 seconds")
    else:
        executable(paths["py2zkvm"], "py2zkvm binary; follow tools/py2zkvm/README.md")
        check(paths["program"].is_file(), f"Missing Python program: {paths['program']}")
        for name in ("cargo", "cargo-prove", "depyler", "rustup"):
            check(shutil.which(name), f"Missing {name}; follow tools/py2zkvm/README.md and set PATH")
        if shutil.which("rustup"):
            result = subprocess.run(["rustup", "toolchain", "list"], capture_output=True, text=True)
            has_rust = result.returncode == 0 and re.search(r"(?m)^1\.94\.0(?:-|\s|$)", result.stdout)
            check(has_rust,
                  "Rust 1.94.0 is required by the generated SP1 project; follow tools/py2zkvm/README.md")
            check(result.returncode == 0 and re.search(r"(?m)^succinct(?:-|\s|$)", result.stdout),
                  "Missing succinct guest toolchain; install SP1 5.2.4 as described in tools/py2zkvm/README.md")
            if has_rust and shutil.which("cargo") and shutil.which("cargo-prove"):
                version = subprocess.run(["cargo", "+1.94.0", "prove", "--version"],
                                         cwd=PY2ZKVM, capture_output=True, text=True)
                compatible = version.returncode == 0 and re.search(r"\b5\.2\.4\b", version.stdout + version.stderr)
                if version.returncode == 0 and not compatible:
                    help_result = subprocess.run(["cargo", "+1.94.0", "prove", "build", "--help"],
                                                 cwd=PY2ZKVM, capture_output=True, text=True)
                    compatible = help_result.returncode == 0 and re.search(
                        r"--tag <TAG>[^\[]*\[default:\s*v5\.2\.4\]", help_result.stdout)
                check(compatible,
                      "SP1 CLI 5.2.4 is required by the generated RV32 project; follow tools/py2zkvm/README.md")
    return errors


def run_command(command, cwd, directory, env=None):
    directory.mkdir(parents=True, exist_ok=True)
    log = directory / "run.log"
    resource = directory / "resources.json"
    measured = [sys.executable, str(ROOT / "scripts/measure_resources.py"),
                "--label", directory.name, "--output", str(resource), "--", *map(str, command)]
    interrupted = False
    with log.open("w") as handle:
        process = subprocess.Popen(measured, cwd=cwd, env={**os.environ, **(env or {})},
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, errors="replace", start_new_session=True)
        try:
            for line in process.stdout:
                handle.write(line)
                handle.flush()
            code = process.wait()
        except KeyboardInterrupt:
            interrupted = True
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            code = 130
    resources = json.loads(resource.read_text()) if resource.exists() else {}
    return {"command": list(map(str, command)), "cwd": str(cwd), "log": str(log),
            "exit_code": code, "wall_seconds": resources.get("wall_seconds"),
            "peak_rss_gib": resources.get("peak_rss_gib"), "interrupted": interrupted}


def match(text, pattern):
    found = re.search(pattern, text, re.MULTILINE)
    if not found:
        raise ValueError(f"Missing result field: {pattern}")
    return found.group(1)


def finite_number(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("Invalid timing or proof size")
    return value


def duration_ms(value):
    found = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)(ns|µs|μs|us|ms|s)", value.strip())
    if not found:
        raise ValueError(f"Unrecognized Rust duration: {value}")
    return finite_number(found[1]) * {"ns": 1e-6, "µs": 1e-3, "μs": 1e-3, "us": 1e-3, "ms": 1, "s": 1000}[found[2]]


def metrics(name, directory, result):
    text = Path(result["log"]).read_text()
    if name == "inference":
        final = text.split("[Final Result]")[-1]
        checks = {"GKR replay": "PASS", "transcript aligned": "YES", "embedding linked": "YES",
                  "lookup verified": "YES", "lookup linked": "YES", "decode verified": "YES", "shift verified": "YES"}
        verified = "[Final Result]" in text and match(final, r"^status:\s*(\S+)") == "PASS"
        verified = verified and all(re.search(r"^" + re.escape(key) + r":\s*" + value + r"\s*$", text, re.MULTILINE)
                                    for key, value in checks.items())
        data = {"verified": bool(verified),
                "prove_ms": finite_number(match(final, r"^prover:\s*([\d.]+) s")) * 1000,
                "verify_ms": finite_number(match(final, r"^verifier:\s*([\d.]+) s")) * 1000,
                "proof_size_bytes": int(match(final, r"^proof_payload_bytes:\s*(\d+)"))}
    elif name == "coral":
        data = {"verified": True,
                "prove_ms": finite_number(match(text, r"^Proving time \(ms\):\s*(\d+)")),
                "verify_ms": finite_number(match(text, r"^Verification time \(ms\):\s*(\d+)")),
                "proof_size_bytes": int(match(text, r"^Proof size \(bytes\):\s*(\d+)"))}
        if (directory / "to_verify.proof").stat().st_size != data["proof_size_bytes"]:
            raise ValueError("Coral proof size does not match its output file")
    elif name == "sp1":
        data = {"verified": True,
                "prove_ms": duration_ms(match(text, r"^Proof generated in (\S+)")),
                "verify_ms": duration_ms(match(text, r"^Proof verified successfully in (\S+)")),
                "proof_size_bytes": (directory / "project/script/proof.bin").stat().st_size,
                "output": match(text, r"^Computation result:\s*(.+)$")}
    else:
        data = json.loads((directory / "result.json").read_text())
        if data.get("component") != "zktls" or type(data.get("verified")) is not bool:
            raise ValueError("Invalid zkTLS result")
        for field in ("prove_ms", "verify_ms"):
            data[field] = finite_number(data[field])
        if type(data.get("proof_size_bytes")) is not int:
            raise ValueError("Invalid zkTLS proof size")
    if data["proof_size_bytes"] <= 0:
        raise ValueError("Empty proof payload")
    data["verified"] = data["verified"] and result["exit_code"] == 0
    return data


def save_summary(output, summary):
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [f"# Table 6: {summary['workload']}", "", f"Status: **{summary['status']}**", "",
             "| Component | Status | Prove (ms) | Verify (ms) | Proof (bytes) | Wall (s) |",
             "|---|---|---:|---:|---:|---:|"]
    for stage in summary["stages"]:
        def cell(key):
            value = stage.get(key)
            return "—" if value is None else f"{value:.6f}" if isinstance(value, float) else str(value)
        lines.append("| " + " | ".join([stage["name"], stage["status"], cell("prove_ms"),
                     cell("verify_ms"), cell("proof_size_bytes"), cell("wall_seconds")]) + " |")
    lines += ["", f"Pipeline wall time: {summary.get('wall_seconds', 0):.3f} s", ""]
    if summary.get("errors"):
        lines += ["## Errors", ""] + [f"- {error}" for error in summary["errors"]] + [""]
    report = "\n".join(lines)
    (output / "summary.md").write_text(report)
    return report


def main(argv=None):
    args = parse_args(argv)
    spec = WORKLOADS[args.workload]
    output = args.output_dir or (ROOT / "benchmark_logs/table6" /
                                f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{args.workload}_{os.getpid()}")
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        print(f"Cannot create new result directory {output}: {error}", file=sys.stderr)
        return 2
    ptau = args.coral_ptau or absolute(os.environ.get("CORAL_PTAU", str(CORAL / "ppot_0080_23.ptau")))
    paths = {"inference": absolute(os.environ.get("BUILD_DIR", "cmake-build-release")) / "src/benchmark_inference",
             "tokens": ROOT / f"demo/{args.workload}_tokens.txt",
             "mask": ROOT / f"demo/{args.workload}_generation_mask.txt",
             "coral": absolute(os.environ.get("CORAL_BIN", str(CORAL / "target/release/coral"))),
             "ptau": ptau, "document": CORAL / "tests/test_docs/json" / spec["document"],
             "grammar": CORAL / "grammars" / spec["grammar"],
             "py2zkvm": absolute(os.environ.get("PY2ZKVM_BIN", str(PY2ZKVM / "py2zkvm/target/release/py2zkvm"))),
             "program": PY2ZKVM / "examples/add.py"}
    names = ["inference"] + (["coral", spec["tool"]] if args.with_tools else [])
    summary = {"workload": args.workload, "lookup_backend": None if args.skip_inference else args.lookup_backend,
               "with_tools": args.with_tools, "skip_inference": args.skip_inference,
               "started_utc": datetime.now(timezone.utc).isoformat(),
               "status": "RUNNING", "inputs": {key: str(value) for key, value in paths.items()},
               "stages": [{"name": name, "status": "SKIPPED" if name == "inference" and args.skip_inference else "NOT_RUN"}
                          for name in names], "errors": []}
    start = time.monotonic()
    exit_code = 1
    try:
        summary["errors"] = preflight(args, paths)
        if args.with_tools and not summary["errors"]:
            try:
                summary["tool_inputs"] = check_tool_inputs(ROOT, args.workload, spec, paths)
            except (OSError, ValueError, SyntaxError, KeyError, TypeError) as error:
                summary["errors"].append(f"Tool input consistency check failed: {error}")
        if summary["errors"]:
            summary["status"] = "BLOCKED"
            for error in summary["errors"]:
                print(error, file=sys.stderr)
            return 2
        inference = [paths["inference"], "--model-dir", args.model_dir, "--token-file", paths["tokens"],
                     "--seq-len", str(spec["length"]), "--generation-mask", paths["mask"],
                     "--decode-batch-size", str(spec["decode_batch"]), "--threads", str(args.threads),
                     "--cache-dir", args.cache_dir, "--use-cache", "--lookup-backend", args.lookup_backend, *args.extra]
        for stage in summary["stages"]:
            name = stage["name"]
            if stage["status"] == "SKIPPED":
                continue
            directory = output / name
            directory.mkdir()
            stage["status"] = "RUNNING"
            save_summary(output, summary)
            env = {}
            if name == "inference":
                command, cwd = inference, ROOT
            elif name == "coral":
                (directory / "ppot_0080_23.ptau").symlink_to(ptau)
                command = [paths["coral"], "--e2e", "-d", paths["document"], "-g", paths["grammar"], "-b", "100"]
                cwd = directory
            elif name == "zktls":
                command, cwd = ["node", "index.js", str(spec["latitude"]), str(spec["longitude"]), spec["city"]], ZKTLS
                env = {"BENCHMARK_RESULT_PATH": str(directory / "result.json")}
                summary["weather_request"] = {"city": spec["city"], "latitude": spec["latitude"],
                                               "longitude": spec["longitude"], "response": "live Open-Meteo response"}
            else:
                project = directory / "project"
                prepare = run_command([paths["py2zkvm"], "transpile", "-i", paths["program"],
                                       "-o", project, "--main-fn", "add"], PY2ZKVM, directory / "prepare")
                stage["preparation"] = prepare
                if prepare["exit_code"]:
                    stage.update(status="FAIL", verified=False)
                    summary["errors"].append("SP1 project generation failed; see sp1/prepare/run.log")
                    exit_code = prepare["exit_code"]
                    break
                command, cwd = ["cargo", "+1.94.0", "run", "--release"], project / "script"
                env = {"CARGO_TARGET_DIR": os.environ.get("CARGO_TARGET_DIR", str(args.cache_dir / "table6-sp1-target"))}
            result = run_command(command, cwd, directory, env)
            stage.update(result)
            try:
                stage.update(metrics(name, directory, result))
                if name == "sp1" and stage["verified"]:
                    if (re.fullmatch(r"-?\d+", stage["output"]) is None
                            or int(stage["output"]) != summary["tool_inputs"]["expected_output"]):
                        raise ValueError("SP1 public output does not match the coding observation")
            except (OSError, ValueError, KeyError, TypeError) as error:
                stage.update(verified=False, error=str(error))
            stage["status"] = "PASS" if stage.get("verified") is True else "FAIL"
            if stage["status"] == "FAIL":
                summary["errors"].append(f"{name} failed: {stage.get('error', 'verification or command failure')}; see {stage['log']}")
                exit_code = result["exit_code"] or 1
                break
        else:
            exit_code = 0
        summary["status"] = "PASS" if exit_code == 0 else "FAIL"
        return exit_code
    except KeyboardInterrupt:
        summary["status"] = "INTERRUPTED"
        return 130
    except (OSError, ValueError) as error:
        summary["status"] = "FAIL"
        summary["errors"].append(str(error))
        print(error, file=sys.stderr)
        return 1
    finally:
        summary["wall_seconds"] = time.monotonic() - start
        for stage in summary["stages"]:
            if stage["status"] == "RUNNING":
                stage.update(status="FAIL", verified=False)
        report = save_summary(output, summary)
        print(report, end="", flush=True)


if __name__ == "__main__":
    sys.exit(main())
