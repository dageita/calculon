from calculon.moe_traffic import (PlacementPolicy, TrafficExperiment,
                                  WaterfillPolicy, scan_strategy_curves,
                                  synthesize_topk_ids)


def test_trace_shape_and_unique_topk():
  trace = synthesize_topk_ids(
      steps=3, tokens_per_step=64, topk=2, routed_experts=8,
      shared_experts=1, hotspot_strength=.8, hotspot_duration=2,
      migration_speed=1, seed=7)
  assert (trace.steps, trace.tokens_per_step, trace.topk) == (3, 64, 2)
  assert all(len(set(row)) == 2 for step in trace.topk_ids for row in step)


def test_waterfill_and_reaction_scan():
  base = dict(steps=8, tokens_per_step=512, topk=1, routed_experts=8,
              shared_experts=1, hotspot_strength=.9,
              hotspot_duration=2, migration_speed=1., seed=3)
  trace = synthesize_topk_ids(**base)
  static = TrafficExperiment(
      trace, num_ranks=4, policy=PlacementPolicy()).run()
  water = TrafficExperiment(
      trace, num_ranks=4, policy=WaterfillPolicy()).run()
  assert water.mean_latency_s <= static.mean_latency_s
  rows = scan_strategy_curves(
      axis="reaction", values=[0, 2], base=base, num_ranks=4)
  assert len(rows) == 2 and rows[1]["controller_reaction_steps"] == 2
