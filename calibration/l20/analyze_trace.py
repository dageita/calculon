#!/usr/bin/env python3
"""Summarize a Kineto trace without fitting or changing simulator parameters."""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def union_duration(intervals):
    total, end = 0.0, float("-inf")
    for left, right in sorted(intervals):
        total += max(0, right-max(left, end))
        end = max(end, right)
    return total


def summarize(path):
    events = json.loads(path.read_text())["traceEvents"]
    gpu = [e for e in events if e.get("cat") in
           ("kernel", "gpu_memcpy", "gpu_memset") and e.get("dur", 0) > 0]
    steps = [e for e in events if e.get("name", "").startswith("ProfilerStep#")
             and e.get("dur", 0) > 0]
    devices = sorted({e.get("args", {}).get("device", e.get("pid")) for e in gpu})
    rows = []
    for step in steps:
        left, right = step["ts"], step["ts"]+step["dur"]
        for device in devices:
            active = [e for e in gpu if
                      e.get("args", {}).get("device", e.get("pid")) == device and
                      e["ts"] < right and e["ts"]+e["dur"] > left]
            intervals = [(max(left, e["ts"]), min(right, e["ts"]+e["dur"])) for e in active]
            busy = union_duration(intervals)
            rows.append(dict(step=step["name"], device=device,
                             wall_s=step["dur"]/1e6, gpu_busy_s=busy/1e6,
                             gpu_uncovered_s=(step["dur"]-busy)/1e6,
                             gpu_events=len(active)))
    grouped = {}
    for category in ("kernel", "cuda_runtime", "cpu_op"):
        durations, counts = defaultdict(float), defaultdict(int)
        for event in events:
            if event.get("cat") == category:
                durations[event["name"]] += event.get("dur", 0)/1e6
                counts[event["name"]] += 1
        grouped[category] = [dict(name=n, inclusive_s=t, calls=counts[n])
                             for n, t in sorted(durations.items(),
                                                key=lambda item: -item[1])[:30]]
    return dict(trace=str(path), steps=rows, top_events=grouped,
                caveats=["Profiling perturbs execution: use unprofiled logs for accuracy.",
                         "GPU busy is interval union, not sum across overlapping streams.",
                         "Uncovered time is not automatically CPU compute: it includes "
                         "host dispatch, synchronization, rank skew and trace boundaries.",
                         "CPU events may be nested; do not sum their inclusive durations."])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(summarize(args.trace), indent=2)
    if args.output:
        args.output.write_text(result+"\n")
    print(result)
