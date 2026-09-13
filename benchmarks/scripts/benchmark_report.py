#!/usr/bin/env python3

import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys

SECTIONS = (
    "Run Configuration", "Preparation", "Circuit Size", "Component Timings",
    "Verification Status", "Final Result",
)
TSV_FIELDS = (
    "profile", "sequence", "prompt", "threads", "lookup", "transcript",
    "cache", "preparation_s", "prover_s", "verifier_s", "proof_payload_bytes", "scope",
    "circuit_and_plans_s", "wall_s", "peak_rss_gib",
)


def read_report(text):
    result = {}
    section = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            if section not in SECTIONS:
                section = None
                continue
            if section in result:
                raise ValueError("duplicate report section: " + section)
            result[section] = {}
        elif section is not None:
            key, separator, value = line.partition(":")
            key, value = key.strip(), value.strip()
            if not separator or not key or not value or key in result[section]:
                raise ValueError("invalid/duplicate report field: " + line)
            result[section][key] = value
    return result


def seconds(report, section, field):
    value = report[section][field]
    if not re.fullmatch(r"[0-9]+\.[0-9]+ s", value):
        raise ValueError("invalid reported time: " + value)
    result = float(value[:-2])
    if not math.isfinite(result) or result < 0:
        raise ValueError("non-finite/negative reported time")
    return result


def require_success(report):
    if tuple(report) != SECTIONS:
        raise ValueError("missing or out-of-order full-run report sections")
    config, preparation, checks, final = (report[key] for key in
        ("Run Configuration", "Preparation", "Verification Status", "Final Result"))
    profile = config["profile"]
    profiles = {
        "table5-zkgpt-standalone": (False, False, False),
        "table6-greedy": (True, True, True),
    }
    if profile not in profiles or config["mode"] != "full" or final["status"] != "PASS":
        raise ValueError("not a successful supported full benchmark")
    linked, embedded, decoded = profiles[profile]
    expected = {
        "GKR replay": "PASS", "lookup verified": "YES",
        "transcript aligned": "YES",
    }
    if linked:
        expected["lookup linked"] = "YES"
    if embedded:
        expected["embedding linked"] = "YES"
    if decoded:
        expected["decode verified"] = "YES"
        expected["shift verified"] = "YES"
    if checks != expected:
        raise ValueError("verification status does not match the requested scope")
    scope = "embedding-ape-complete-gkr-lookup" if embedded else "transformer-components"
    if decoded:
        scope += "-greedy-shift"
    if final["scope"] != scope or config["table"] != ("6" if embedded else "5"):
        raise ValueError("report scope/table mismatch")
    if config["transcript"] != "Fiat-Shamir":
        raise ValueError("report transcript mode mismatch")
    if set(preparation) not in ({"total"}, {"total", "circuit_and_plans"}):
        raise ValueError("unexpected preparation fields")
    seconds(report, "Preparation", "total")
    if "circuit_and_plans" in preparation:
        seconds(report, "Preparation", "circuit_and_plans")
    seconds(report, "Final Result", "prover")
    seconds(report, "Final Result", "verifier")
    if not re.fullmatch(r"[0-9]+", final["proof_payload_bytes"]):
        raise ValueError("invalid proof payload count")
    return report


def main():
    parser = argparse.ArgumentParser(description='Read the same short English report shown to artifact reviewers.')
    parser.add_argument("log", nargs="?", type=Path)
    parser.add_argument("--header", action="store_true")
    parser.add_argument("--resources", type=Path)
    parser.add_argument("--expect", action="append", default=[], metavar="FIELD=VALUE")
    args = parser.parse_args()
    writer = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
    if args.header:
        writer.writerow(TSV_FIELDS)
        return
    if args.log is None:
        parser.error("provide a report log or --header")
    report = require_success(read_report(args.log.read_text()))
    config, final = report["Run Configuration"], report["Final Result"]
    for item in args.expect:
        key, separator, value = item.partition("=")
        if not separator or config.get(key) != value:
            raise ValueError("benchmark configuration mismatch: " + item)
    row = {key: config[key] for key in TSV_FIELDS[:7]}
    row.update(preparation_s=seconds(report, SECTIONS[1], "total"),
               prover_s=seconds(report, "Final Result", "prover"),
               verifier_s=seconds(report, "Final Result", "verifier"),
               proof_payload_bytes=final["proof_payload_bytes"], scope=final["scope"])
    row["circuit_and_plans_s"] = (
        seconds(report, "Preparation", "circuit_and_plans")
        if "circuit_and_plans" in report["Preparation"] else "N/A")
    row.update(wall_s="N/A", peak_rss_gib="N/A")
    if args.resources:
        resources = json.loads(args.resources.read_text())
        if resources["exit_code"] != 0 or resources["status"] != "COMPLETED":
            raise ValueError("resource record is not a completed successful command")
        for field, source in (("wall_s", "wall_seconds"), ("peak_rss_gib", "peak_rss_gib")):
            value = resources[source]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("invalid resource measurement: " + source)
            row[field] = value
    writer.writerow(row[key] for key in TSV_FIELDS)


if __name__ == "__main__":
    try:
        main()
    except (KeyError, OSError, ValueError) as error:
        sys.exit("Invalid benchmark report: " + str(error))
