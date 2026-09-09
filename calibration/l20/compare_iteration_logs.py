#!/usr/bin/env python3
"""Extract steady-state Megatron iteration time for simulator comparison."""
import argparse
import json
import re
from pathlib import Path
from statistics import mean, median, pstdev

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--log", type=Path, required=True)
parser.add_argument("--simulator-json", type=Path)
parser.add_argument("--warmup", type=int, default=5)
parser.add_argument("--output", type=Path)
args = parser.parse_args()
iteration_pattern = re.compile(
    r"^\s*\[[^\n]+\] iteration\s+(\d+)\s*/\s*\d+\s*\|"
    r"(?P<body>.*?)(?=^\s*\[[^\n]+\] iteration\s+|\Z)",
    re.MULTILINE | re.DOTALL)
elapsed_pattern = re.compile(r"elapsed time per iteration \(ms\):\s*([0-9.]+)")
skipped_pattern = re.compile(r"number of skipped iterations:\s*(\d+)")
records = []
for match in iteration_pattern.finditer(args.log.read_text(errors="ignore")):
  elapsed = elapsed_pattern.search(match.group(0))
  if not elapsed:
    continue
  skipped = skipped_pattern.search(match.group(0))
  records.append({
      "iteration": int(match.group(1)),
      "elapsed_s": float(elapsed.group(1)) / 1000,
      # Older Megatron logs may not emit this counter; preserve compatibility by
      # treating the missing field as a completed update.
      "skipped": int(skipped.group(1)) if skipped else 0,
      "timers_s": {name: {"min": float(lo)/1000, "max": float(hi)/1000}
                   for name, lo, hi in re.findall(
                     r"^\s+([a-z][a-z0-9-]+)\s+\.+:\s*\(([0-9.]+),\s*([0-9.]+)\)",
                     match.group(0), re.MULTILINE)},
  })
# Compatibility fallback for logs that predate Megatron's per-iteration header.
if not records:
  records = [{"iteration": index + 1, "elapsed_s": float(value) / 1000,
              "skipped": 0}
             for index, value in enumerate(elapsed_pattern.findall(
                 args.log.read_text(errors="ignore")))]
drop = args.warmup if len(records) > args.warmup else (1 if len(records) > 1 else 0)
steady_records = records[drop:]
# A skipped FP16 update does not execute Megatron's normal optimizer path.  The
# simulator models a completed training step, so including it would compare two
# different workloads.  Keep the count visible in the report rather than hiding
# this fact behind an averaged timing.
completed_records = [record for record in steady_records if record["skipped"] == 0]
steady = [record["elapsed_s"] for record in completed_records]
if steady_records and not steady:
  raise SystemExit("all post-warmup Megatron iterations skipped optimizer updates")
result = {"model": args.model, "samples": len(steady),
          "total_post_warmup_samples": len(steady_records),
          "skipped_update_samples": len(steady_records) - len(steady),
          "warmup_samples_dropped": drop,
          "skipped_updates_excluded": True,
          "megatron_iteration_time_s": mean(steady) if steady else None}
if steady:
  trim_count = int(len(steady) * 0.1)
  ordered = sorted(steady)
  trimmed = ordered[trim_count:len(ordered)-trim_count] if trim_count else ordered
  result["iteration_distribution_s"] = {
      "min": min(steady), "median": median(steady), "max": max(steady),
      "stddev": pstdev(steady),
      "coefficient_of_variation": pstdev(steady) / mean(steady),
      "trimmed_mean_10pct": mean(trimmed),
      "trimmed_stddev": pstdev(trimmed),
      "trimmed_coefficient_of_variation": pstdev(trimmed) / mean(trimmed),
      "trimmed_samples": len(trimmed)}
  result["completed_iterations"] = [r["iteration"] for r in completed_records]
  names = sorted({n for r in completed_records for n in r.get("timers_s", {})})
  result["megatron_stage_times_s"] = {
      n: {side: mean(r["timers_s"][n][side] for r in completed_records
                     if n in r.get("timers_s", {})) for side in ("min", "max")}
      for n in names}
  result["measurement_warnings"] = []
  if result["skipped_update_samples"]:
    result["measurement_warnings"].append(
      "Dynamic loss scaling is still skipping updates after warmup.")
  if pstdev(steady)/mean(steady) > 0.1:
    result["measurement_warnings"].append(
      "Iteration coefficient of variation exceeds 10%; mean is not a stable latency.")
  result["timer_interpretation"] = (
    "Stage min/max, when present, are across ranks, may be nested and must "
    "not be summed. Enabling detailed timers synchronizes CUDA and perturbs "
    "execution; the default reference run uses timing-log-level 0.")
if args.simulator_json:
  cv = result.get("iteration_distribution_s", {}).get(
      "trimmed_coefficient_of_variation")
  invalid_reasons = []
  if len(steady) < 3:
    invalid_reasons.append("fewer than three completed post-warmup iterations")
  if cv is not None and cv > 0.1:
    invalid_reasons.append("10% trimmed iteration coefficient of variation exceeds 10%")
  result["accuracy_stability_metric"] = "10pct_trimmed_coefficient_of_variation"
  result["measurement_valid_for_accuracy"] = not invalid_reasons
  result["measurement_invalid_reasons"] = invalid_reasons
  sim = json.loads(args.simulator_json.read_text())
  predicted = sim["summary"]["batch_total_time"]
  measured = result["iteration_distribution_s"]["trimmed_mean_10pct"]
  if measured is None or measured <= 0:
    raise SystemExit("Megatron iteration time is missing or non-positive")
  error_s = predicted - measured
  signed = error_s / measured
  signed_percent = signed * 100.0
  # Keep relative_error as a fraction for old callers, but expose explicit
  # percentage fields so renderers cannot label 0.515 as 0.515%.
  measured_median = result["iteration_distribution_s"]["median"]
  measured_trimmed = result["iteration_distribution_s"]["trimmed_mean_10pct"]
  result.update(
      simulator_batch_total_time_s=predicted,
      comparison_megatron_iteration_time_s=measured,
      comparison_statistic="10% trimmed mean of completed post-warmup iterations",
      error_s=error_s,
      relative_error=signed,
      relative_error_fraction=signed,
      relative_error_percent=signed_percent,
      absolute_percentage_error=abs(signed_percent),
      median_relative_error_percent=(predicted / measured_median - 1) * 100,
      trimmed_mean_relative_error_percent=(predicted / measured_trimmed - 1) * 100,
      error_formula="(simulator_batch_total_time_s - comparison_megatron_iteration_time_s) / comparison_megatron_iteration_time_s",
      error_input_consistent=abs(
          signed - (predicted - measured) / measured) < 1e-12,
  )
if args.simulator_json and not result.get("measurement_valid_for_accuracy", False):
  result.update(
      comparison_valid=False,
      error_s=None,
      relative_error=None,
      relative_error_fraction=None,
      relative_error_percent=None,
      absolute_percentage_error=None,
      median_relative_error_percent=None,
      trimmed_mean_relative_error_percent=None,
      error_formula=None,
      error_input_consistent=None,
  )
elif args.simulator_json:
  result["comparison_valid"] = True
rendered = json.dumps(result, indent=2)
print(rendered)
if args.output:
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(rendered + "\n")
if args.simulator_json and not result.get("measurement_valid_for_accuracy", False):
  raise SystemExit("Megatron measurement is not stable enough for an accuracy comparison: "
                   + "; ".join(result.get("measurement_invalid_reasons", [])))
