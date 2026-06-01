"""Detect anomalous duration increases in test duration files.

Compares new durations against a baseline (old) set and flags tests whose
duration increased by more than a configurable factor.  Produces a Markdown
report and exits with code 1 when anomalies are found.

Usage (from the repo root):
    python .github/scripts/detect_duration_anomalies.py \
        --old-dir repo/.github/durations \
        --new-dir durations \
        --report anomaly_report.md
"""

import argparse
import json
import os
import sys

# Thresholds
RATIO_THRESHOLD = 10  # flag if new/old exceeds this factor
MAX_DISPLAY = 20  # cap table rows per file

DURATION_FILES = ["core_tests_durations.json", "jax_tests_durations.json"]


def find_anomalies(old_data, new_data):
    """Return a list of (test, old_dur, new_dur, ratio) tuples for anomalous tests."""
    inflated = []
    for test, new_dur in new_data.items():
        old_dur = old_data.get(test, 0)
        ratio = new_dur / old_dur
        if ratio > RATIO_THRESHOLD:
            inflated.append((test, old_dur, new_dur, ratio))
    inflated.sort(key=lambda x: -x[3])
    return inflated


def build_report(anomalies_by_file):
    """Build a Markdown report string from anomaly results."""
    lines = []
    for fname, inflated in anomalies_by_file.items():
        if not inflated:
            continue
        lines.append(f"### {fname}")
        lines.append(
            f"**{len(inflated)} tests with >{RATIO_THRESHOLD}x duration increase "
        )
        lines.append("| Test | Old | New | Ratio |")
        lines.append("|------|-----|-----|-------|")
        for test, od, nd, r in inflated[:MAX_DISPLAY]:
            short = test.split("::")[-1][:60]
            lines.append(f"| `{short}` | {od:.1f}s | {nd:.1f}s | **{r:.0f}x** |")
        if len(inflated) > MAX_DISPLAY:
            lines.append(f"| ... and {len(inflated) - MAX_DISPLAY} more | | | |")

    if lines:
        header = (
            "## \u26a0\ufe0f Duration Anomalies Detected\n\n"
            "The following tests show suspicious duration increases, "
            "which may indicate CI contention rather than real slowdowns. "
            "Review before merging.\n\n"
        )
        return header + "\n".join(lines)

    return "No duration anomalies detected."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-dir", required=True, help="Directory with baseline duration JSONs")
    parser.add_argument("--new-dir", required=True, help="Directory with new duration JSONs")
    parser.add_argument("--report", default="anomaly_report.md", help="Output report path")
    args = parser.parse_args()

    anomalies_by_file = {}
    for fname in DURATION_FILES:
        old_path = os.path.join(args.old_dir, fname)
        new_path = os.path.join(args.new_dir, fname)
        if not os.path.exists(old_path) or not os.path.exists(new_path):
            continue
        with open(old_path) as f:
            old_data = json.load(f)
        with open(new_path) as f:
            new_data = json.load(f)
        anomalies_by_file[fname] = find_anomalies(old_data, new_data)

    report = build_report(anomalies_by_file)
    with open(args.report, "w") as f:
        f.write(report)

    has_anomalies = any(anomalies_by_file.values())

    # Write GITHUB_OUTPUT if running in CI
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"has_anomalies={'true' if has_anomalies else 'false'}\n")

    if has_anomalies:
        sys.exit(1)


if __name__ == "__main__":
    main()
