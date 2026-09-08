"""Synthetic top-k traffic traces and a common load-balance experiment machine."""
from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from statistics import mean
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class TrafficTrace:
  """Logical router output with shape [step][token][topk]."""
  topk_ids: tuple
  routed_experts: int
  shared_experts: int = 0
  step_duration_s: float = 1.0
  metadata: Mapping[str, object] = field(default_factory=dict)

  def __post_init__(self):
    if self.routed_experts <= 0 or self.shared_experts < 0:
      raise ValueError("invalid expert count")
    if self.step_duration_s <= 0:
      raise ValueError("step_duration_s must be positive")
    for step in self.topk_ids:
      for choices in step:
        if not choices or len(set(choices)) != len(choices):
          raise ValueError("top-k choices must be non-empty and unique")
        if min(choices) < 0 or max(choices) >= self.routed_experts:
          raise ValueError("top-k id outside routed expert range")

  @property
  def steps(self):
    return len(self.topk_ids)

  @property
  def tokens_per_step(self):
    return len(self.topk_ids[0]) if self.topk_ids else 0

  @property
  def topk(self):
    return len(self.topk_ids[0][0]) if self.tokens_per_step else 0

  def expert_counts(self, step):
    counts = [0] * self.routed_experts
    for choices in self.topk_ids[step]:
      for expert in choices:
        counts[expert] += 1
    return counts


def synthesize_topk_ids(*, steps, tokens_per_step, topk, routed_experts,
                        shared_experts=0, hotspot_strength=0.0,
                        hotspot_duration=1, migration_speed=0.0,
                        step_duration_s=1.0, seed=1234):
  """Generate logical top-k ids with a persistent, migrating hot expert."""
  if steps <= 0 or tokens_per_step <= 0:
    raise ValueError("steps and tokens_per_step must be positive")
  if not 1 <= topk <= routed_experts:
    raise ValueError("topk outside routed expert range")
  if not 0 <= hotspot_strength <= 1:
    raise ValueError("hotspot_strength must be in [0, 1]")
  if hotspot_duration <= 0 or migration_speed < 0:
    raise ValueError("invalid hotspot duration or migration speed")
  rng = random.Random(seed)
  result = []
  for step in range(steps):
    hot = int((step // hotspot_duration) * migration_speed) % routed_experts
    rows = []
    for _ in range(tokens_per_step):
      first = hot if rng.random() < hotspot_strength else rng.randrange(routed_experts)
      choices = [first]
      while len(choices) < topk:
        candidate = rng.randrange(routed_experts)
        if candidate not in choices:
          choices.append(candidate)
      rows.append(tuple(choices))
    result.append(tuple(rows))
  return TrafficTrace(
      tuple(result), routed_experts, shared_experts, step_duration_s,
      {"hotspot_strength": hotspot_strength,
       "hotspot_duration": hotspot_duration,
       "migration_speed": migration_speed, "seed": seed})


class ExperimentState(str, Enum):
  WARMUP = "warmup"
  OBSERVE = "observe"
  PLAN = "plan"
  APPLY = "apply"
  MEASURE = "measure"
  DONE = "done"


class PlacementPolicy:
  name = "static"

  def plan(self, observed: Sequence[float], num_ranks: int,
           current: Sequence[int]):
    return list(current)


class WaterfillPolicy(PlacementPolicy):
  """Greedily place routed experts; shared experts remain replicated."""
  name = "waterfill"

  def plan(self, observed, num_ranks, current):
    rank_load = [0.0] * num_ranks
    placement = [0] * len(observed)
    for expert in sorted(range(len(observed)), key=lambda e: (-observed[e], e)):
      target = min(range(num_ranks), key=lambda rank: (rank_load[rank], rank))
      placement[expert] = target
      rank_load[target] += observed[expert]
    return placement


@dataclass(frozen=True)
class ExperimentMetrics:
  policy: str
  mean_latency_s: float
  p95_latency_s: float
  throughput_tokens_s: float
  imbalance_area: float
  step_latency_s: tuple
  transitions: tuple


class TrafficExperiment:
  """Unified state machine used by static and waterfill strategies."""
  def __init__(self, trace, *, num_ranks, policy,
               controller_reaction_steps=0, routed_service_tokens_s=1.0,
               shared_service_cost=1.0):
    if num_ranks <= 0 or controller_reaction_steps < 0:
      raise ValueError("invalid ranks or reaction steps")
    self.trace, self.num_ranks, self.policy = trace, num_ranks, policy
    self.reaction = controller_reaction_steps
    self.rate, self.shared_cost = routed_service_tokens_s, shared_service_cost

  def run(self):
    placement = [e % self.num_ranks for e in range(self.trace.routed_experts)]
    pending, transitions, latencies = [], [], []
    area = 0.0
    for step in range(self.trace.steps):
      transitions.append((step, ExperimentState.OBSERVE.value))
      counts = self.trace.expert_counts(step)
      transitions.append((step, ExperimentState.PLAN.value))
      pending.append((step + self.reaction,
                      self.policy.plan(counts, self.num_ranks, placement)))
      applicable = [item for item in pending if item[0] <= step]
      if applicable:
        transitions.append((step, ExperimentState.APPLY.value))
        placement = applicable[-1][1]
        pending = [item for item in pending if item[0] > step]
      shared = self.trace.shared_experts * self.trace.tokens_per_step
      loads = [shared * self.shared_cost for _ in range(self.num_ranks)]
      for expert, count in enumerate(counts):
        loads[placement[expert]] += count
      average = sum(loads) / self.num_ranks
      area += (max(loads) - average) * self.trace.step_duration_s
      latencies.append(max(loads) / self.rate)
      transitions.append((step, ExperimentState.MEASURE.value))
    transitions.append((self.trace.steps, ExperimentState.DONE.value))
    ordered = sorted(latencies)
    p95 = ordered[max(0, math.ceil(.95 * len(ordered)) - 1)]
    total_tokens = self.trace.steps * self.trace.tokens_per_step
    return ExperimentMetrics(
        self.policy.name, mean(latencies), p95,
        total_tokens / sum(latencies), area, tuple(latencies),
        tuple(transitions))


def scan_strategy_curves(*, axis: str, values: Iterable[float], base: Mapping,
                         num_ranks: int, reaction_steps=0):
  """Return hotspot->latency/throughput gain or reaction->imbalance curves."""
  keys = {"strength": "hotspot_strength", "duration": "hotspot_duration",
          "migration_speed": "migration_speed", "reaction": None}
  if axis not in keys:
    raise ValueError("unknown scan axis: " + axis)
  rows = []
  for value in values:
    kwargs, reaction = dict(base), reaction_steps
    if axis == "reaction":
      reaction = int(value)
    else:
      kwargs[keys[axis]] = int(value) if axis == "duration" else float(value)
    trace = synthesize_topk_ids(**kwargs)
    static = TrafficExperiment(
        trace, num_ranks=num_ranks, policy=PlacementPolicy()).run()
    water = TrafficExperiment(
        trace, num_ranks=num_ranks, policy=WaterfillPolicy(),
        controller_reaction_steps=reaction).run()
    rows.append({
        "axis": axis, "value": value,
        "static_latency_s": static.mean_latency_s,
        "waterfill_latency_s": water.mean_latency_s,
        "latency_gain": static.mean_latency_s / water.mean_latency_s,
        "static_throughput_tokens_s": static.throughput_tokens_s,
        "waterfill_throughput_tokens_s": water.throughput_tokens_s,
        "throughput_gain": water.throughput_tokens_s / static.throughput_tokens_s,
        "imbalance_area": water.imbalance_area,
        "controller_reaction_steps": reaction})
  return rows


def main(argv=None):
  parser = argparse.ArgumentParser()
  parser.add_argument("--axis", choices=("strength", "duration",
                      "migration_speed", "reaction"), required=True)
  parser.add_argument("--values", required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--steps", type=int, default=32)
  parser.add_argument("--tokens", type=int, default=4096)
  parser.add_argument("--topk", type=int, default=2)
  parser.add_argument("--routed-experts", type=int, default=16)
  parser.add_argument("--shared-experts", type=int, default=1)
  parser.add_argument("--ranks", type=int, default=4)
  args = parser.parse_args(argv)
  base = dict(steps=args.steps, tokens_per_step=args.tokens, topk=args.topk,
              routed_experts=args.routed_experts,
              shared_experts=args.shared_experts,
              hotspot_strength=.5, hotspot_duration=4, migration_speed=1.)
  rows = scan_strategy_curves(
      axis=args.axis, values=[float(x) for x in args.values.split(",")],
      base=base, num_ranks=args.ranks)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  with args.output.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
  main()
