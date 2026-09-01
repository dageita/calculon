"""Constrained 5D parallelism and superpod hardware/software co-design search."""

from __future__ import annotations

import copy
import itertools
import logging
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from calculon import System
from calculon.llm.llm import Llm
from calculon.llm.runner import Runner


DIMENSIONS = ("tp", "cp", "ep", "pp", "dp")
DEFAULT_DENSE_POLICIES = ("tp-cp-pp-dp-ep", "tp-pp-cp-dp-ep")
DEFAULT_MOE_POLICIES = (
    "tp-ep-cp-pp-dp",
    "tp-cp-ep-pp-dp",
    "ep-tp-cp-pp-dp",
)


def factors(value: int) -> Iterable[int]:
    """Yield positive factors without walking all integers up to ``value``."""
    low, high = [], []
    candidate = 1
    while candidate * candidate <= value:
        if value % candidate == 0:
            low.append(candidate)
            if candidate * candidate != value:
                high.append(value // candidate)
        candidate += 1
    yield from low
    yield from reversed(high)


def _radical_inverse(index: int, base: int) -> float:
    """Deterministic low-discrepancy coordinate in ``[0, 1)``."""
    result = 0.0
    fraction = 1.0 / base
    while index:
        result += fraction * (index % base)
        index //= base
        fraction /= base
    return result


def _unique(values: Sequence) -> List:
    return list(dict.fromkeys(values))


def _option(config, name: str, default: Sequence) -> List:
    value = getattr(config, name, None)
    return _unique(value if value else default)


def validate_objective(config) -> Tuple[Tuple[str, ...], Optional[float]]:
    raw = getattr(config, "objectives", None)
    if not raw:
        raw = (getattr(config, "objective", None) or "throughput",)
    objectives = tuple(_unique([str(value).lower() for value in raw]))
    allowed = {"throughput", "batch_time", "mfu", "time_to_train"}
    unknown = set(objectives) - allowed
    if unknown:
        raise Llm.Error("unsupported search objectives: " + ", ".join(sorted(unknown)))
    if not objectives:
        raise Llm.Error("at least one search objective is required")
    training_samples = getattr(config, "training_samples", None)
    if "time_to_train" in objectives and (training_samples is None or training_samples <= 0):
        raise Llm.Error("training_samples must be > 0 for time_to_train")
    return objectives, training_samples


def fixed_global_batch_size(config) -> int:
    """Return the configured fixed GBS (legacy Optimal Mode behavior)."""
    batch = getattr(config, "global_batch_size", None)
    if batch is None:
        batch = getattr(config, "max_global_batch_size", None)
    if batch is None:
        batch = getattr(config, "max_batch_size", None)
    if batch is None or int(batch) <= 0:
        raise Llm.Error("global_batch_size (or max_batch_size) must be > 0")
    return int(batch)


def _attention_type(app: Llm.Application) -> str:
    return "mla" if app.is_mla else "multihead"


def generate_5d_parallelisms(app: Llm.Application, num_procs: int,
                             global_batch_size: int,
                             search_up_to: bool = False) -> List[Dict[str, int]]:
    """Generate legal 5D and batch/microbatch combinations."""
    if num_procs <= 0:
        raise Llm.Error("num_procs must be > 0")
    candidates: List[Dict[str, int]] = []
    for tp in factors(num_procs):
        if app.hidden % tp or app.attn_heads % tp or app.kv_heads % tp:
            continue
        after_tp = num_procs // tp
        for pp in factors(after_tp):
            if pp > app.num_blocks or app.num_blocks % pp:
                continue
            after_pp = after_tp // pp
            ep_values = factors(after_pp) if app.is_moe else (1,)
            for ep in ep_values:
                if app.is_moe and app.num_experts % ep:
                    continue
                after_ep = after_pp // ep
                for cp in factors(after_ep):
                    if app.seq_size % cp:
                        continue
                    dp = after_ep // cp
                    batch_values = (range(dp, global_batch_size + 1, dp)
                                    if search_up_to else (global_batch_size,))
                    for batch_size in batch_values:
                        if batch_size % dp:
                            continue
                        local_batch = batch_size // dp
                        for ppint in Llm.get_valid_pipeline_interleavings(
                                app.num_blocks, pp):
                            for microbatch in factors(local_batch):
                                if microbatch * app.seq_size % tp:
                                    continue
                                candidates.append({
                                    "tp": tp, "pp": pp, "dp": dp, "ep": ep,
                                    "cp": cp, "pipeline_interleaving": ppint,
                                    "batch_size": batch_size,
                                    "microbatch_size": microbatch,
                                })
    return candidates


@dataclass(frozen=True)
class GroupPlacement:
    dimension: str
    degree: int
    stride: int
    local_fraction: float
    remote_fraction: float
    network_tier: int
    effective_bandwidth: float
    effective_latency: float

    def as_dict(self) -> Dict:
        return {
            "degree": self.degree,
            "stride": self.stride,
            "local_fraction": self.local_fraction,
            "remote_fraction": self.remote_fraction,
            "network_tier": self.network_tier,
            "effective_bandwidth": self.effective_bandwidth,
            "effective_latency": self.effective_latency,
        }


@dataclass(frozen=True)
class PlacementResult:
    policy: str
    scale_up_size: int
    groups: Dict[str, GroupPlacement]

    @property
    def network_ids(self) -> Dict[str, int]:
        return {name: placement.network_tier for name, placement in self.groups.items()}

    def as_dict(self) -> Dict:
        return {
            "policy": self.policy,
            "scale_up_size": self.scale_up_size,
            "groups": {name: item.as_dict() for name, item in self.groups.items()},
        }


class PlacementPlanner:
    """Map hierarchical rank order to local/remote collective fractions.

    Ranks are contiguous in the policy order, from inner to outer dimension.
    Mixed local/remote communication is represented by a serial-path harmonic bandwidth.
    """

    @staticmethod
    def normalize_policy(policy: str) -> Tuple[str, ...]:
        dims = tuple(part.strip().lower() for part in policy.split("-") if part.strip())
        if set(dims) != set(DIMENSIONS) or len(dims) != len(DIMENSIONS):
            raise Llm.Error(
                f"placement policy must contain {DIMENSIONS} exactly once: {policy}")
        return dims

    @staticmethod
    def plan(degrees: Dict[str, int], policy: str, scale_up_size: int,
             intra_bw: float, inter_bw: float, intra_latency: float,
             inter_latency: float) -> PlacementResult:
        dims = PlacementPlanner.normalize_policy(policy)
        if scale_up_size <= 0:
            raise Llm.Error("scale_up_size must be > 0")
        if intra_bw <= 0 or inter_bw <= 0:
            raise Llm.Error("intra/inter bandwidth must be > 0")

        raw: Dict[str, Tuple[int, int, float]] = {}
        stride = 1
        for dim in dims:
            degree = int(degrees[dim])
            if degree <= 1:
                remote = 0.0
            else:
                local_peers = max(1, min(degree, scale_up_size // stride))
                remote = 1.0 - (local_peers - 1) / (degree - 1)
                remote = min(1.0, max(0.0, remote))
            raw[dim] = (degree, stride, remote)
            stride *= degree

        groups: Dict[str, GroupPlacement] = {}
        for dim in DIMENSIONS:
            degree, dim_stride, remote = raw[dim]
            local = 1.0 - remote
            inverse_bw = local / intra_bw + remote / inter_bw
            effective_bw = 1.0 / inverse_bw
            effective_latency = local * intra_latency + remote * inter_latency
            groups[dim] = GroupPlacement(
                dimension=dim, degree=degree, stride=dim_stride,
                local_fraction=local, remote_fraction=remote,
                network_tier=1 if remote > 0 else 0,
                effective_bandwidth=effective_bw,
                effective_latency=effective_latency,
            )
        return PlacementResult(policy, scale_up_size, groups)


def build_candidate_system(base_json: Dict, hardware: Dict,
                           placement: PlacementResult, logger) -> System:
    cfg = copy.deepcopy(base_json)
    networks = cfg.get("networks") or []
    if not networks:
        raise Llm.Error("system networks configuration is empty")
    while len(networks) < 2:
        networks.append(copy.deepcopy(networks[0]))
    networks[0]["bandwidth"] = float(hardware["intra_bandwidth"])
    networks[0]["latency"] = float(hardware["intra_latency"])
    networks[0]["size"] = int(hardware["scale_up_size"])
    networks[1]["bandwidth"] = float(hardware["inter_bandwidth"])
    networks[1]["latency"] = float(hardware["inter_latency"])
    networks[1]["size"] = max(int(networks[1].get("size") or 0), int(hardware["num_procs"]))
    for network in networks[:2]:
        network["topology"] = hardware["network_topology"]
        network["flow_parameters"] = {
            dim: {
                "bandwidth": placement.groups[dim].effective_bandwidth,
                "latency": placement.groups[dim].effective_latency,
            }
            for dim in DIMENSIONS
        }
    return System(cfg, logger)


def _parallel_score(candidate: Dict, scale_up_size: int) -> Tuple:
    """Cheap ordering only; exact ranking always comes from the simulator."""
    inner_pressure = candidate["tp"] * candidate["cp"] * candidate["ep"]
    overflow = max(0, inner_pressure - scale_up_size)
    pipeline_penalty = candidate["pp"] / max(1, candidate["pipeline_interleaving"])
    return (overflow, -(candidate["tp"] * candidate["pp"] * candidate["ep"]), pipeline_penalty, -candidate["batch_size"], -candidate["microbatch_size"],
            candidate["dp"], candidate["tp"])


def expand_software_candidates(app: Llm.Application, config,
                               scale_up_size: int) -> Tuple[List[Dict], int]:
    num_procs = int(config.num_procs)
    max_global_batch = getattr(config, "max_global_batch_size", None)
    search_batch_sizes = max_global_batch is not None
    global_batch = (int(max_global_batch) if search_batch_sizes
                    else fixed_global_batch_size(config))
    bases = generate_5d_parallelisms(app, num_procs, global_batch, search_batch_sizes)
    recomputes = _option(config, "activation_recompute_options",
                         ("full", "attn_only", "none"))
    shardings = _option(config, "optimizer_sharding_options", (False, True))
    comm_types = _option(config, "tensor_par_comm_types",
                         ("ar", "p2p_rs_ag", "rs_ag"))
    tp_overlaps = _option(config, "tensor_par_overlap_options", ("none",))
    dp_overlaps = _option(config, "data_par_overlap_options", (False, True))
    weight_offloads = _option(config, "weight_offload_options", (False,))
    activation_offloads = _option(
        config, "activations_offload_options", (False,))
    optimizer_offloads = _option(
        config, "optimizer_offload_options", (False,))
    expanded: List[Dict] = []
    for base in bases:
        options = itertools.product(
            recomputes, shardings, comm_types, tp_overlaps, dp_overlaps,
            weight_offloads, activation_offloads, optimizer_offloads)
        for (recompute, sharding, comm_type, tp_overlap, dp_overlap,
             weight_offload, activations_offload,
             optimizer_offload) in options:
            if sharding and base["dp"] <= 1:
                continue
            if dp_overlap and base["dp"] <= 1:
                continue
            if tp_overlap != "none" and base["tp"] <= 1:
                continue
            if tp_overlap != "none" and (app.is_mla or app.ffn_type == "swiglu"):
                continue
            item = dict(base)
            if activations_offload and recompute == "full":
                continue
            item.update({
                "activation_recompute": recompute,
                "optimizer_sharding": sharding,
                "tensor_par_comm_type": comm_type,
                "tensor_par_overlap": tp_overlap,
                "data_par_overlap": dp_overlap,
                "weight_offload": weight_offload,
                "activations_offload": activations_offload,
                "optimizer_offload": optimizer_offload,
            })
            expanded.append(item)
    generated = len(expanded)
    expanded.sort(key=lambda item: _parallel_score(item, scale_up_size))
    limit = max(1, int(getattr(config, "max_candidates", None) or 512))
    coarse_top_k = max(1, int(getattr(config, "coarse_top_k", None) or limit))
    flow_limit = min(limit, coarse_top_k)
    primary = expanded[:flow_limit]
    primary_ids = {id(item) for item in primary}
    fallback = [item for item in expanded if id(item) not in primary_ids]
    # max_global_batch_size is an upper bound, not a fixed batch.  Candidates
    # rejected by the memory preflight do not consume the expensive flow budget;
    # order the remaining candidates by memory friendliness so an OOM at the
    # upper bound can fall back through smaller microbatches/batches.
    recompute_rank = {"full": 0, "attn_only": 1, "none": 2}
    fallback.sort(key=lambda item: (
        -item["batch_size"], item["microbatch_size"],
        recompute_rank.get(item["activation_recompute"], 3),
        not item["optimizer_sharding"],
        _parallel_score(item, scale_up_size)))
    fallback_limit = max(
        1, int(getattr(config, "oom_fallback_candidates", None) or 256))
    # Keep a memory-friendly representative for every batch first, then for
    # every (batch, microbatch) pair, and finally fill the bounded preflight
    # queue.  This preserves the max-bound fallback without retaining the full
    # combinatorial expansion (which can exceed 100k candidates).
    selected_fallback = []
    selected_ids = set()
    for key_fn in (
            lambda item: item["batch_size"],
            lambda item: (item["batch_size"], item["microbatch_size"])):
        seen_keys = set()
        for item in fallback:
            key = key_fn(item)
            if key in seen_keys or id(item) in selected_ids:
                continue
            seen_keys.add(key)
            selected_ids.add(id(item))
            selected_fallback.append(item)
            if len(selected_fallback) >= fallback_limit:
                break
        if len(selected_fallback) >= fallback_limit:
            break
    if len(selected_fallback) < fallback_limit:
        for item in fallback:
            if id(item) in selected_ids:
                continue
            selected_fallback.append(item)
            if len(selected_fallback) >= fallback_limit:
                break
    return primary + selected_fallback, generated


def make_execution_json(app: Llm.Application, config, candidate: Dict,
                        placement: PlacementResult) -> Dict:
    matrix_dtype = getattr(config, "matrix_dtype", None) or getattr(config, "datatype", None)
    vector_dtype = getattr(config, "vector_dtype", None) or matrix_dtype
    nets = placement.network_ids
    return {
        "num_procs": int(config.num_procs),
        "tensor_par": candidate["tp"],
        "pipeline_par": candidate["pp"],
        "data_par": candidate["dp"],
        "expert_par": candidate["ep"],
        "context_par": candidate["cp"],
        "tensor_par_net": nets["tp"],
        "pipeline_par_net": nets["pp"],
        "data_par_net": nets["dp"],
        "expert_par_net": nets["ep"],
        "context_par_net": nets["cp"],
        "batch_size": candidate["batch_size"],
        "microbatch_size": candidate["microbatch_size"],
        "datatype": matrix_dtype,
        "matrix_dtype": matrix_dtype,
        "vector_dtype": vector_dtype,
        "fused_activation": True,
        "attention_type": _attention_type(app),
        "activation_recompute": candidate["activation_recompute"],
        "pipeline_interleaving": candidate["pipeline_interleaving"],
        "optimizer_sharding": candidate["optimizer_sharding"],
        "tensor_par_comm_type": candidate["tensor_par_comm_type"],
        "tensor_par_overlap": candidate["tensor_par_overlap"],
        "seq_par_ag_redo": False,
        "data_par_overlap": candidate["data_par_overlap"],
        "weight_offload": candidate["weight_offload"],
        "activations_offload": candidate["activations_offload"],
        "optimizer_offload": candidate["optimizer_offload"],
        "training": True,
    }


def _public_execution(execution: Dict) -> Dict:
    mapping = {
        "num_procs": "gpu_numbers", "tensor_par": "tensor_parallel",
        "pipeline_par": "pipeline_parallel", "data_par": "data_parallel",
        "expert_par": "expert_parallel", "context_par": "context_parallel",
        "tensor_par_comm_type": "tensor_parallel_comm_type",
        "tensor_par_overlap": "tensor_parallel_overlap",
        "seq_par_ag_redo": "sequence_parallel_allgather_redo",
        "data_par_overlap": "data_parallel_overlap",
    }
    hidden = {"tensor_par_net", "pipeline_par_net", "data_par_net",
              "expert_par_net", "context_par_net"}
    return {mapping.get(key, key): value for key, value in execution.items()
            if key not in hidden}


OBJECTIVE_FIELDS = {
    "throughput": ("linear_scaling_throughput", True),
    "batch_time": ("batch_total_time", False),
    "mfu": ("total_efficiency", True),
    "time_to_train": ("time_to_train_seconds", False),
    "gpu_cost": ("gpu_resource_cost", False),
}


def _rank_metric_items(items: List[Dict], objectives: Sequence[str]) -> List[Dict]:
    """Pareto-rank candidates and choose a balanced compromise.

    Each selected metric is min-max normalized.  Multi-objective candidates on
    the Pareto front are ordered by minimax loss first, then mean loss, avoiding
    a hidden preference caused by incomparable physical units.
    """
    if not items:
        return []
    ranges = {}
    for objective in objectives:
        field, _ = OBJECTIVE_FIELDS[objective]
        values = [float(item["metrics"][field]) for item in items]
        ranges[objective] = (min(values), max(values))
    for item in items:
        losses = {}
        for objective in objectives:
            field, maximize = OBJECTIVE_FIELDS[objective]
            value = float(item["metrics"][field])
            low, high = ranges[objective]
            if high <= low:
                loss = 0.0
            elif maximize:
                loss = (high - value) / (high - low)
            else:
                loss = (value - low) / (high - low)
            losses[objective] = loss
        item["objective_losses"] = losses
        item["balanced_score"] = max(losses.values()) + sum(losses.values()) / len(losses)

    def dominates(left: Dict, right: Dict) -> bool:
        left_losses = left["objective_losses"]
        right_losses = right["objective_losses"]
        return (all(left_losses[key] <= right_losses[key] for key in objectives)
                and any(left_losses[key] < right_losses[key] for key in objectives))

    for item in items:
        item["pareto_optimal"] = not any(
            other is not item and dominates(other, item) for other in items)
    return sorted(items, key=lambda item: (
        not item["pareto_optimal"], item["balanced_score"],
        sum(item["objective_losses"].values())))


HARDWARE_IDENTITY_FIELDS = (
    "num_procs", "scale_up_size", "scale_out_size", "intra_bandwidth",
    "inter_pod_bandwidth", "intra_latency", "inter_latency",
    "network_topology",
)


def _hardware_identity(hardware: Dict) -> Tuple:
    return tuple(hardware.get(key) for key in HARDWARE_IDENTITY_FIELDS)


def _software_comparison_signature(candidate: Dict) -> Tuple:
    """Identify an identical workload/software strategy across hardware points."""
    parallel = candidate.get("parallelism") or {}
    return tuple(sorted(parallel.items())) + (
        ("placement_policy", (candidate.get("placement") or {}).get("policy")),
    )


def _normalized_hardware_investment(hardware: Dict,
                                    ranges: Dict[str, Tuple[float, float]]) -> float:
    """Estimate relative design investment without assuming monetary prices."""
    costs = []
    for key in ("num_procs", "scale_up_size", "intra_bandwidth", "inter_pod_bandwidth"):
        low, high = ranges[key]
        value = float(hardware[key])
        costs.append(0.0 if high <= low else (value - low) / (high - low))
    for key in ("intra_latency", "inter_latency"):
        low, high = ranges[key]
        value = float(hardware[key])
        costs.append(0.0 if high <= low else (high - value) / (high - low))
    return sum(costs) / len(costs)


def _rank_hardware_fair(evaluated_designs: List[Dict],
                        candidate_records: Sequence[Dict],
                        objectives: Sequence[str]) -> Tuple[List[Dict], int, Dict]:
    """Rank hardware globally with one workload and grouped normalized regret."""
    if not evaluated_designs:
        return [], 0, {}
    hardware_ids = {
        _hardware_identity(item["result"]["hardware"])
        for item in evaluated_designs
    }
    batch_coverage: Dict[int, set] = {}
    for candidate in candidate_records:
        batch = int((candidate.get("parallelism") or {}).get("batch_size") or 0)
        if batch <= 0:
            continue
        identity = _hardware_identity(candidate.get("hardware") or {})
        if identity in hardware_ids:
            batch_coverage.setdefault(batch, set()).add(identity)
    if not batch_coverage:
        ranked = _rank_metric_items(evaluated_designs, objectives)
        return ranked, 0, {"method": "legacy-no-common-batch"}
    comparison_batch = max(
        batch_coverage,
        key=lambda batch: (len(batch_coverage[batch]), batch),
    )
    comparable = [
        candidate for candidate in candidate_records
        if int((candidate.get("parallelism") or {}).get("batch_size") or 0)
        == comparison_batch
    ]
    candidates_by_hardware: Dict[Tuple, List[Dict]] = {}
    for candidate in comparable:
        candidates_by_hardware.setdefault(
            _hardware_identity(candidate["hardware"]), []).append(candidate)
    # Representative metrics used by knee detection also come from the common
    # batch, rather than each point's earlier local result.
    for item in evaluated_designs:
        identity = _hardware_identity(item["result"]["hardware"])
        local_candidates = candidates_by_hardware.get(identity, [])
        if local_candidates:
            local_ranked = _rank_metric_items(local_candidates, objectives)
            item["metrics"] = dict(local_ranked[0]["metrics"])

    signature_groups: Dict[Tuple, List[Dict]] = {}
    for candidate in comparable:
        signature_groups.setdefault(
            _software_comparison_signature(candidate), []).append(candidate)

    regrets: Dict[Tuple, List[float]] = {}
    for group in signature_groups.values():
        for candidate in _rank_metric_items(group, objectives):
            identity = _hardware_identity(candidate["hardware"])
            regrets.setdefault(identity, []).append(
                min(1.0, float(candidate["balanced_score"]) / 2.0))

    range_keys = (
        "num_procs", "scale_up_size", "intra_bandwidth", "inter_pod_bandwidth",
        "intra_latency", "inter_latency",
    )
    ranges = {
        key: (
            min(float(item["result"]["hardware"][key]) for item in evaluated_designs),
            max(float(item["result"]["hardware"][key]) for item in evaluated_designs),
        )
        for key in range_keys
    }
    max_coverage = max((len(values) for values in regrets.values()), default=1)
    for item in evaluated_designs:
        hardware = item["result"]["hardware"]
        identity = _hardware_identity(hardware)
        values = regrets.get(identity, [])
        mean_regret = sum(values) / len(values) if values else 1.0
        worst_regret = max(values) if values else 1.0
        coverage_loss = 1.0 - len(values) / max_coverage
        investment = _normalized_hardware_investment(hardware, ranges)
        # Hardware investment is not an implicit optimization objective. The user
        # selected performance metrics determine the optimum; investment only
        # breaks ties between performance-equivalent designs.
        components = (mean_regret, worst_regret, coverage_loss)
        item["balanced_score"] = max(components) + sum(components) / len(components)
        item["objective_losses"] = {
            "mean_comparable_regret": mean_regret,
            "worst_comparable_regret": worst_regret,
            "coverage_loss": coverage_loss,
        }
        item["design_investment"] = investment
        item["comparison_signature_coverage"] = len(values)

    def dominates(left: Dict, right: Dict) -> bool:
        keys = left["objective_losses"]
        return (
            all(left["objective_losses"][key] <= right["objective_losses"][key]
                for key in keys)
            and any(left["objective_losses"][key] < right["objective_losses"][key]
                    for key in keys)
        )

    for item in evaluated_designs:
        item["pareto_optimal"] = not any(
            other is not item and dominates(other, item)
            for other in evaluated_designs
        )
    ranked = sorted(evaluated_designs, key=lambda item: (
        not item["pareto_optimal"], item["balanced_score"],
        sum(item["objective_losses"].values()), item["design_investment"]))
    details = {
        "method": "common-batch grouped normalized robust regret",
        "comparison_batch_size": comparison_batch,
        "comparable_software_signatures": len(signature_groups),
        "hardware_coverage": len(batch_coverage[comparison_batch]),
        "hardware_points": len(hardware_ids),
        "includes_normalized_design_investment": False,
        "hardware_investment_role": "tie_break_only",
    }
    return ranked, comparison_batch, details


def _simulate_cached_profile(system: System, execution: Dict,
                             profile: Dict) -> Tuple:
    """Run only the hardware-dependent flow layer for a cached software profile."""
    nets = {
        "tp": execution["tensor_par_net"],
        "cp": execution["context_par_net"],
        "ep": execution["expert_par_net"],
        "pp": execution["pipeline_par_net"],
        "dp": execution["data_par_net"],
    }
    fabrics = {name: system.get_network(tier) for name, tier in nets.items()}
    topology_fabric = fabrics["dp"]
    if execution["data_par"] <= 1 and execution["expert_par"] > 1:
        topology_fabric = fabrics["ep"]
    elif execution["data_par"] <= 1 and execution["pipeline_par"] > 1:
        topology_fabric = fabrics["pp"]
    flow = system.get_network(0)
    flow.flow_network_init(
        tp_bw=fabrics["tp"].flow_bandwidth("tp"),
        cp_bw=fabrics["cp"].flow_bandwidth("cp"),
        ep_bw=fabrics["ep"].flow_bandwidth("ep"),
        pp_bw=fabrics["pp"].flow_bandwidth("pp"),
        dp_bw=fabrics["dp"].flow_bandwidth("dp"),
        topology=topology_fabric._topology,
        tp_latency=fabrics["tp"].flow_latency("tp"),
        cp_latency=fabrics["cp"].flow_latency("cp"),
        ep_latency=fabrics["ep"].flow_latency("ep"),
        pp_latency=fabrics["pp"].flow_latency("pp"),
        dp_latency=fabrics["dp"].flow_latency("dp"))
    kwargs = dict(profile["flow_kwargs"])
    kwargs["enable_timeline"] = False
    return flow.total_flow_network_time(**kwargs)


class ConstrainedSearch:
    """Shared exact evaluator for Optimal Mode and Superpod Mode."""

    @staticmethod
    def run(logger, app: Llm.Application, config, hardware: Dict,
            system_factory: Callable[[PlacementResult], System],
            placement_policies: Optional[Sequence[str]] = None,
            software_candidates: Optional[Tuple[List[Dict], int]] = None,
            profile_cache: Optional[Dict] = None) -> Dict:
        started = time.monotonic()
        objectives, training_samples = validate_objective(config)
        policies = tuple(placement_policies or getattr(config, "placement_policies", None)
                         or (DEFAULT_MOE_POLICIES if app.is_moe else DEFAULT_DENSE_POLICIES))
        software, generated = (software_candidates or
                               expand_software_candidates(
                                   app, config, int(hardware["scale_up_size"])))
        profile_cache = profile_cache if profile_cache is not None else {}
        design_logger = logging.getLogger("design-search")
        design_logger.setLevel(logging.INFO)
        logger.info("Design-search hardware point: scale_up=%s candidates=%d policies=%d",
                    hardware["scale_up_size"], len(software), len(policies))
        top_n = max(1, int(getattr(config, "top_n", None) or 1))
        flow_limit = min(
            max(1, int(getattr(config, "max_candidates", None) or 512)),
            max(1, int(getattr(config, "coarse_top_k", None) or 512)))
        evaluated_items: List[Dict] = []
        good = bad = oom = duplicate_placements = evaluated = 0
        profile_cache_hits = profile_cache_misses = 0
        errors: Dict[str, int] = {}
        search_candidates: List[Dict] = []
        for candidate in software:
            if good >= flow_limit:
                break
            degrees = {name: candidate[name] for name in DIMENSIONS}
            seen_placements = set()
            for policy in policies:
                if good >= flow_limit:
                    break
                try:
                    placement = PlacementPlanner.plan(
                        degrees, policy, int(hardware["scale_up_size"]),
                        float(hardware["intra_bandwidth"]) * float(hardware.get("intra_efficiency", 1.0)),
                        float(hardware["inter_bandwidth"]) * float(hardware.get("inter_efficiency", 1.0)),
                        float(hardware["intra_latency"]),
                        float(hardware["inter_latency"]))
                    signature = _placement_signature(placement)
                    if signature in seen_placements:
                        duplicate_placements += 1
                        continue
                    seen_placements.add(signature)
                    evaluated += 1
                    if evaluated == 1 or evaluated % 25 == 0:
                        logger.info(
                            "Design-search progress: evaluated=%d/%d TP=%d PP=%d DP=%d EP=%d CP=%d batch=%d microbatch=%d",
                            evaluated, len(software) * len(policies), candidate["tp"], candidate["pp"],
                            candidate["dp"], candidate["ep"], candidate["cp"], candidate["batch_size"], candidate["microbatch_size"])
                    profile_key = tuple(sorted(
                        (key, value) for key, value in candidate.items()))
                    cached_profile = profile_cache.get(profile_key)
                    if cached_profile and cached_profile["oom"]:
                        profile_cache_hits += 1
                        oom += 1
                        continue
                    profile_cache_misses += cached_profile is None
                    profile_cache_hits += cached_profile is not None
                    system = system_factory(placement)
                    execution = make_execution_json(app, config, candidate, placement)
                    model = None
                    if cached_profile is None:
                        model = Llm(app, design_logger)
                        model.compile(system, Llm.Execution.from_json(execution))
                        model.run(system)
                        over_capacity = model.mem_over_capacity()
                        if over_capacity:
                            profile_cache[profile_key] = {"oom": True}
                            oom += 1
                            continue
                        global_time = float(model.get_flow_network_global_time())
                        perfect_time = model.get_total_efficiency() * global_time
                        flow_kwargs = dict(model._flow_network_kwargs(False))
                        profile_cache[profile_key] = {
                            "oom": False,
                            "perfect_time": perfect_time,
                            "flow_kwargs": flow_kwargs,
                            "compute": {key: flow_kwargs[key] for key in (
                                "fwdCompTime", "bwdCompTime", "fwd_mla_time",
                                "fwd_ffn_time", "bwd_mla_time", "bwd_ffn_time")},
                            "communication_volumes": {
                                key: value for key, value in flow_kwargs.items()
                                if key.lower().endswith("size")},
                            "memory": {
                                "tier1_required_bytes": model.get_mem_tier1_cap_req(),
                                "tier2_required_bytes": model.get_mem_tier2_cap_req(),
                            },
                            "memory_over_capacity": False,
                        }
                        cached_profile = profile_cache[profile_key]
                        total_comm_time = float(model.get_flow_network_total_comm_time())
                    else:
                        flow_result = _simulate_cached_profile(
                            system, execution, cached_profile)
                        global_time = float(flow_result[0])
                        total_comm_time = float(flow_result[12])
                        perfect_time = float(cached_profile["perfect_time"])
                    throughput = float(execution["batch_size"]) / global_time
                    if throughput <= 0:
                        raise Llm.Error("simulator returned non-positive throughput")
                    good += 1
                    candidate_metrics = {
                        "batch_total_time": global_time,
                        "linear_scaling_throughput": throughput,
                        "total_efficiency": round(min(1.0, perfect_time / global_time), 6),
                        "total_comm_time": total_comm_time,
                        "time_to_train_seconds": (
                            float(training_samples) / throughput
                            if training_samples is not None else None),
                    }
                    candidate_record = {
                        "candidate_id": len(search_candidates) + 1,
                        "hardware": {
                            key: hardware.get(key) for key in (
                                "num_procs", "max_gpu_numbers", "scale_up_size", "scale_out_size",
                                "intra_bandwidth", "inter_bandwidth", "inter_pod_bandwidth",
                                "intra_latency", "inter_latency", "network_topology",
                                "sweep_dimension", "chart_controlled",
                                "chart_context_rank",
                                "chart_series_context")
                        },
                        "parallelism": _public_execution(execution),
                        "placement": placement.as_dict(),
                        "metrics": candidate_metrics,
                    }
                    if hasattr(config, "scale_up_sizes"):
                        search_candidates.append(candidate_record)
                    evaluated_items.append({
                        "metrics": candidate_metrics,
                        "execution": execution,
                        "placement": placement.as_dict(),
                        "placement_object": placement,
                        "model": model,
                        "candidate_record": candidate_record,
                        "software_candidate": candidate,
                    })
                except Exception as exc:  # invalid combinations are pruned here
                    bad += 1
                    key = str(exc).split("\n", 1)[0][:240]
                    errors[key] = errors.get(key, 0) + 1

        elapsed = max(time.monotonic() - started, 1e-9)
        executions = {
            "generated_candidates": generated * len(policies),
            "pruned_by_flow_limit": max(0, generated - good) * len(policies),
            "total_executions": good + bad + oom,
            "good_executions": good,
            "bad_executions": bad,
            "oom_executions": oom,
            "duplicate_placements_pruned": duplicate_placements,
            "calculation_rate": (good + bad + oom) / elapsed,
            "stage1_legal_candidates": generated,
            "stage2_flow_candidate_limit": flow_limit,
            "stage2_flow_candidates": good,
            "oom_fallback_candidates_tried": oom,
            "profile_cache_hits": profile_cache_hits,
            "profile_cache_misses": profile_cache_misses,
        }
        ranked_items = _rank_metric_items(evaluated_items, objectives)
        for item in ranked_items:
            item["candidate_record"]["optimization"] = {
                "pareto_optimal": item["pareto_optimal"],
                "balanced_score": item["balanced_score"],
                "objective_losses": item["objective_losses"],
            }
        best = ranked_items[:top_n]
        if not best:
            return {"status": "error", "error": "No acceptable configurations found",
                    "executions": executions, "error_summary": errors}

        results = []
        for item in best:
            execution = item["execution"]
            placement = item["placement"]
            model = item["model"]
            if model is None:
                system = system_factory(item["placement_object"])
                model = Llm(app, design_logger)
                model.compile(system, Llm.Execution.from_json(execution))
                model.run(system)
            detailed = Runner.get_simulator_res_json(model)
            summary = dict(detailed["summary"])
            summary["objectives"] = list(objectives)
            summary["objective"] = objectives[0]
            summary["objective_value"] = item["balanced_score"]
            summary["pareto_optimal"] = item["pareto_optimal"]
            summary["objective_losses"] = item["objective_losses"]
            summary["total_comm_time"] = item["metrics"]["total_comm_time"]
            throughput = float(summary["linear_scaling_throughput"])
            summary["throughput_samples_per_second"] = throughput
            if training_samples is not None:
                summary["training_samples"] = float(training_samples)
                summary["time_to_train_seconds"] = float(training_samples) / throughput
            public_hardware = {
                **{key: hardware.get(key) for key in (
                    "scale_up_size", "scale_out_size", "intra_bandwidth", "inter_bandwidth",
                    "inter_pod_bandwidth", "intra_latency", "inter_latency",
                    "network_topology")},
                "gpu_numbers": int(execution["num_procs"]),
                # Compatibility aliases now both carry the exact fixed count.
                "selected_gpu_numbers": int(execution["num_procs"]),
                "max_gpu_numbers": int(execution["num_procs"]),
            }
            results.append({
                "optimal_result": _public_execution(execution),
                "hardware": public_hardware,
                "placement": placement,
                "memory_usage": detailed["memory_usage"],
                "computation": detailed["computation"],
                "communication": detailed["communication"],
                "summary": summary,
            })
        response = dict(results[0])
        response.update({
            "executions": executions,
            "objective": objectives[0],
            "objectives": list(objectives),
            "search_candidates": search_candidates,
            "top_results": results,
            "error_summary": errors,
            "_ranked_software_candidates": [
                item["software_candidate"] for item in ranked_items],
        })
        return response


def hardware_points(config, defaults: Dict) -> Iterable[Dict]:
    """Yield hardware points for one fixed GPU count and legal scale-up factors."""
    fixed_num_procs = int(config.num_procs)
    if fixed_num_procs <= 0:
        raise Llm.Error("num_procs (GPU Numbers) must be > 0")

    # GPU Numbers is fixed. Legacy fields may only repeat that exact count.
    explicit_gpu_numbers = getattr(config, "gpu_numbers", None)
    max_num_procs = getattr(config, "max_num_procs", None)
    if explicit_gpu_numbers:
        requested = sorted(_unique(int(value) for value in explicit_gpu_numbers))
        if requested != [fixed_num_procs]:
            raise Llm.Error(
                "gpu_numbers is no longer a search dimension; it must equal num_procs")
    if max_num_procs is not None and int(max_num_procs) != fixed_num_procs:
        raise Llm.Error(
            "max_num_procs is deprecated; GPU Numbers is fixed by num_procs")
    gpu_numbers = [fixed_num_procs]

    explicit_scale_ups = getattr(config, "scale_up_sizes", None)
    max_scale_up = getattr(config, "max_scale_up_size", None)
    if max_scale_up is not None and int(max_scale_up) <= 0:
        raise Llm.Error("max_scale_up_size must be > 0")

    inter_pod_bandwidths = getattr(config, "inter_pod_bandwidths", None)
    inter_field = "inter_pod_bandwidth" if inter_pod_bandwidths else "inter_bandwidth"
    inter_values = (inter_pod_bandwidths or getattr(config, "inter_bandwidths", None)
                    or [defaults["inter_bandwidth"]])
    common_fields = {
        "intra_bandwidth": getattr(config, "intra_bandwidths", None)
                             or [defaults["intra_bandwidth"]],
        inter_field: inter_values,
        "intra_latency": getattr(config, "intra_latencies", None)
                           or [defaults["intra_latency"]],
        "inter_latency": getattr(config, "inter_latencies", None)
                           or [defaults["inter_latency"]],
        "network_topology": getattr(config, "network_topologies", None)
                              or [defaults["network_topology"]],
    }
    common_fields = {key: _unique(values) for key, values in common_fields.items()}
    sampling = (getattr(config, "hardware_sampling", None) or "cartesian").lower()
    if sampling not in ("cartesian", "one_at_a_time", "balanced"):
        raise Llm.Error("hardware_sampling must be cartesian, one_at_a_time, or balanced")

    raw_points = []
    for num_procs in gpu_numbers:
        if explicit_scale_ups:
            scale_up_sizes = [int(value) for value in explicit_scale_ups
                              if int(value) > 0 and int(value) <= num_procs
                              and num_procs % int(value) == 0]
        elif max_scale_up is not None:
            cap = min(num_procs, int(max_scale_up))
            scale_up_sizes = [value for value in factors(num_procs) if value <= cap]
        else:
            catalog_size = min(num_procs, int(defaults["scale_up_size"]))
            scale_up_sizes = [catalog_size] if num_procs % catalog_size == 0 else [
                max(value for value in factors(num_procs) if value <= catalog_size)]
        if not scale_up_sizes:
            continue
        fields = {"scale_up_size": _unique(scale_up_sizes), **common_fields}
        if sampling == "cartesian":
            for values in itertools.product(*fields.values()):
                raw_points.append({
                    **dict(zip(fields.keys(), values)),
                    "num_procs": num_procs,
                    "max_gpu_numbers": fixed_num_procs,
                    "sweep_dimension": "joint",
                })
        else:
            baseline = {"scale_up_size": max(fields["scale_up_size"])}
            for key, values in common_fields.items():
                if key == "inter_pod_bandwidth":
                    default_pod_bandwidth = (
                        float(defaults["inter_bandwidth"]) * baseline["scale_up_size"])
                    baseline[key] = min(
                        values, key=lambda value: abs(float(value) - default_pod_bandwidth))
                elif key in defaults and isinstance(defaults[key], (int, float)):
                    baseline[key] = min(
                        values, key=lambda value: abs(float(value) - float(defaults[key])))
                else:
                    baseline[key] = values[0]
            base = {
                **baseline,
                "num_procs": num_procs,
                "max_gpu_numbers": fixed_num_procs,
                "sweep_dimension": "baseline",
            }
            raw_points.append(base)
            for key, values in fields.items():
                for value in values:
                    raw_points.append({
                        **base, key: value, "sweep_dimension": key,
                    })
            if sampling == "balanced":
                # Preserve interpretable axis sweeps and add joint points so
                # bandwidth/latency/scale-up interactions are considered.
                joint_samples = max(
                    0, int(getattr(config, "global_hardware_samples", None) or 12))
                field_keys = list(fields)
                primes = (2, 3, 5, 7, 11, 13, 17)
                corners = (
                    {
                        key: (min(values) if "latency" in key else max(values))
                        for key, values in fields.items()
                    },
                    {
                        key: (max(values) if "latency" in key else min(values))
                        for key, values in fields.items()
                    },
                )
                for values in corners:
                    raw_points.append({
                        **base, **values, "sweep_dimension": "joint",
                    })
                for sample in range(1, joint_samples + 1):
                    values = {}
                    for position, key in enumerate(field_keys):
                        choices = fields[key]
                        coordinate = _radical_inverse(sample, primes[position])
                        index = min(
                            len(choices) - 1,
                            int(coordinate * len(choices)),
                        )
                        values[key] = choices[index]
                    raw_points.append({
                        **base, **values, "sweep_dimension": "joint",
                    })

    seen = set()
    identity_keys = (
        "num_procs", "scale_up_size", "intra_bandwidth", inter_field,
        "intra_latency", "inter_latency", "network_topology")
    for point in raw_points:
        num_procs = int(point["num_procs"])
        scale_up_size = int(point["scale_up_size"])
        if scale_up_size <= 0 or num_procs % scale_up_size:
            continue
        point["scale_out_size"] = num_procs // scale_up_size
        if "inter_pod_bandwidth" in point:
            point["inter_pod_bandwidth"] = float(point["inter_pod_bandwidth"])
            point["inter_bandwidth"] = point["inter_pod_bandwidth"] / scale_up_size
        else:
            point["inter_bandwidth"] = float(point["inter_bandwidth"])
            point["inter_pod_bandwidth"] = point["inter_bandwidth"] * scale_up_size
        if float(point["intra_bandwidth"]) <= 0 or float(point["inter_bandwidth"]) <= 0:
            continue
        if float(point["intra_latency"]) < 0 or float(point["inter_latency"]) < 0:
            continue
        identity = tuple(point[key] for key in identity_keys)
        if identity in seen:
            continue
        seen.add(identity)
        point["intra_efficiency"] = float(defaults.get("intra_efficiency", 1.0))
        point["inter_efficiency"] = float(defaults.get("inter_efficiency", 1.0))
        yield point


def _summary_metrics(summary: Dict) -> Dict:
    return {
        "batch_total_time": float(summary["batch_total_time"]),
        "linear_scaling_throughput": float(summary["linear_scaling_throughput"]),
        "total_efficiency": float(summary["total_efficiency"]),
        "total_comm_time": float(summary["total_comm_time"]),
        "time_to_train_seconds": summary.get("time_to_train_seconds"),
    }


def _bandwidth_latency_knees(designs: Sequence[Dict]) -> Dict:
    """Return 95%-of-maximum-throughput knees for continuous fabric inputs."""
    specs = {
        "intra_bandwidth": ("Intra-node Bandwidth", "GB/s", True),
        "inter_pod_bandwidth": ("Inter-node Bandwidth", "GB/s per pod", True),
        "intra_latency": ("Intra-node Latency", "s", False),
        "inter_latency": ("Inter-node Latency", "s", False),
    }
    knees = {}
    for key, (label, unit, bandwidth_like) in specs.items():
        buckets: Dict[float, float] = {}
        for entry in designs:
            hardware = entry["result"]["hardware"]
            if hardware.get(key) is None:
                continue
            x = float(hardware[key])
            throughput = float(entry["metrics"]["linear_scaling_throughput"])
            buckets[x] = max(buckets.get(x, 0.0), throughput)
        if not buckets:
            continue
        peak = max(buckets.values())
        acceptable = [x for x, value in buckets.items() if value >= 0.95 * peak]
        # Smallest sufficient BW is the design knee; largest tolerable latency
        # is the latency knee.
        knee = min(acceptable) if bandwidth_like else max(acceptable)
        knees[key] = {
            "label": label,
            "value": knee,
            "unit": unit,
            "throughput_at_knee": buckets[knee],
            "peak_throughput": peak,
            "criterion": "at least 95% of peak throughput",
        }
    return knees


def _adaptive_hardware_points(coarse_points: Sequence[Dict], designs: Sequence[Dict],
                              limit: int) -> List[Dict]:
    """Bisect coarse intervals adjacent to each measured 95% knee."""
    if not designs or limit <= 0:
        return []
    knees = _bandwidth_latency_knees(designs)
    ranked = _rank_metric_items(list(designs), tuple(
        designs[0]["result"].get("objectives") or ["throughput"]))
    base = dict(ranked[0]["result"]["hardware"])
    existing = {
        (point["num_procs"], point["scale_up_size"], point["intra_bandwidth"],
         point["inter_pod_bandwidth"], point["intra_latency"],
         point["inter_latency"], point["network_topology"])
        for point in coarse_points
    }
    refined = []
    for key in ("intra_bandwidth", "inter_pod_bandwidth",
                "intra_latency", "inter_latency"):
        values = sorted({float(point[key]) for point in coarse_points})
        if len(values) < 2 or key not in knees:
            continue
        knee = float(knees[key]["value"])
        index = min(range(len(values)), key=lambda pos: abs(values[pos] - knee))
        neighbors = []
        if index > 0:
            neighbors.append((values[index - 1] + values[index]) / 2.0)
        if index + 1 < len(values):
            neighbors.append((values[index] + values[index + 1]) / 2.0)
        for value in neighbors:
            point = dict(base)
            point[key] = value
            point["sweep_dimension"] = f"adaptive:{key}"
            if key == "inter_pod_bandwidth":
                point["inter_bandwidth"] = value / int(point["scale_up_size"])
            identity = (
                point["num_procs"], point["scale_up_size"], point["intra_bandwidth"],
                point["inter_pod_bandwidth"], point["intra_latency"],
                point["inter_latency"], point["network_topology"])
            if identity not in existing:
                existing.add(identity)
                refined.append(point)
                if len(refined) >= limit:
                    return refined
    return refined


def _raw_execution_from_result(result: Dict) -> Dict:
    """Recreate an Execution JSON from a public optimal result."""
    public = result["optimal_result"]
    placement_groups = (result.get("placement") or {}).get("groups") or {}
    inverse = {
        "gpu_numbers": "num_procs",
        "tensor_parallel": "tensor_par",
        "pipeline_parallel": "pipeline_par",
        "data_parallel": "data_par",
        "expert_parallel": "expert_par",
        "context_parallel": "context_par",
        "tensor_parallel_comm_type": "tensor_par_comm_type",
        "tensor_parallel_overlap": "tensor_par_overlap",
        "sequence_parallel_allgather_redo": "seq_par_ag_redo",
        "data_parallel_overlap": "data_par_overlap",
    }
    execution = {inverse.get(key, key): value for key, value in public.items()}
    network_fields = {
        "tp": "tensor_par_net",
        "cp": "context_par_net",
        "ep": "expert_par_net",
        "pp": "pipeline_par_net",
        "dp": "data_par_net",
    }
    for dimension, field in network_fields.items():
        execution[field] = int(
            (placement_groups.get(dimension) or {}).get("network_tier", 0))
    return execution


def _gpu_design_points(base_system_json: Dict, config) -> Tuple[List[Dict], Dict]:
    """Generate nearby GPU compute/capacity/HBM design points."""
    matrix_dtype = getattr(config, "matrix_dtype", None) or getattr(
        config, "datatype", None)
    matrix_cfg = (base_system_json.get("matrix") or {}).get(matrix_dtype)
    if matrix_cfg is None:
        raise Llm.Error(f"matrix dtype {matrix_dtype!r} is unavailable for GPU design")
    baseline = {
        "matrix_dtype": matrix_dtype,
        "compute_tflops": float(matrix_cfg["tflops"]),
        "memory_capacity_gib": float(base_system_json["mem1"]["GiB"]),
        "memory_bandwidth_gbps": float(base_system_json["mem1"]["GBps"]),
    }
    factor_fields = {
        "compute_factor": _option(
            config, "gpu_compute_factors", (0.8, 1.0, 1.2)),
        "memory_capacity_factor": _option(
            config, "gpu_memory_capacity_factors", (0.8, 1.0, 1.2)),
        "memory_bandwidth_factor": _option(
            config, "gpu_memory_bandwidth_factors", (0.8, 1.0, 1.2)),
    }
    if any(float(value) <= 0 for values in factor_fields.values() for value in values):
        raise Llm.Error("GPU design factors must all be > 0")
    points = []
    for compute_factor, capacity_factor, bandwidth_factor in itertools.product(
            factor_fields["compute_factor"],
            factor_fields["memory_capacity_factor"],
            factor_fields["memory_bandwidth_factor"]):
        points.append({
            "matrix_dtype": matrix_dtype,
            "compute_tflops": baseline["compute_tflops"] * float(compute_factor),
            "memory_capacity_gib": (
                baseline["memory_capacity_gib"] * float(capacity_factor)),
            "memory_bandwidth_gbps": (
                baseline["memory_bandwidth_gbps"] * float(bandwidth_factor)),
            "compute_factor": float(compute_factor),
            "memory_capacity_factor": float(capacity_factor),
            "memory_bandwidth_factor": float(bandwidth_factor),
        })
    return points, baseline


def _search_gpu_design(logger, app: Llm.Application, base_system_json: Dict,
                       config, fixed_result: Dict, objectives: Sequence[str],
                       training_samples: Optional[float]) -> Tuple[Dict, List[Dict]]:
    """Search GPU specifications while holding hardware and software fixed."""
    points, baseline = _gpu_design_points(base_system_json, config)
    hardware = fixed_result["hardware"]
    execution = _raw_execution_from_result(fixed_result)
    parallel = fixed_result["optimal_result"]
    degrees = {
        "tp": int(parallel["tensor_parallel"]),
        "pp": int(parallel["pipeline_parallel"]),
        "dp": int(parallel["data_parallel"]),
        "ep": int(parallel["expert_parallel"]),
        "cp": int(parallel["context_parallel"]),
    }
    policy = (fixed_result.get("placement") or {}).get("policy")
    placement = PlacementPlanner.plan(
        degrees, policy, int(hardware["scale_up_size"]),
        float(hardware["intra_bandwidth"]) * float(
            hardware.get("intra_efficiency", 1.0)),
        float(hardware["inter_bandwidth"]) * float(
            hardware.get("inter_efficiency", 1.0)),
        float(hardware["intra_latency"]),
        float(hardware["inter_latency"]),
    )
    evaluated = []
    public_candidates = []
    for point in points:
        candidate_json = copy.deepcopy(base_system_json)
        dtype = point["matrix_dtype"]
        candidate_json["matrix"][dtype]["tflops"] = point["compute_tflops"]
        candidate_json["mem1"]["GiB"] = point["memory_capacity_gib"]
        candidate_json["mem1"]["GBps"] = point["memory_bandwidth_gbps"]
        try:
            system = build_candidate_system(
                candidate_json, hardware, placement, logger)
            model = Llm(app, logging.getLogger("gpu-design-search"))
            model.compile(system, Llm.Execution.from_json(execution))
            model.run(system)
            if model.mem_over_capacity():
                continue
            detailed = Runner.get_simulator_res_json(model)
            candidate_summary = dict(detailed["summary"])
            candidate_summary["total_comm_time"] = float(
                model.get_flow_network_total_comm_time())
            metrics = _summary_metrics(candidate_summary)
            throughput = float(metrics["linear_scaling_throughput"])
            if training_samples is not None:
                metrics["time_to_train_seconds"] = float(training_samples) / throughput
            metrics["gpu_resource_cost"] = (
                point["compute_factor"] + point["memory_capacity_factor"]
                + point["memory_bandwidth_factor"]
            ) / 3.0
            item = {
                "metrics": metrics,
                "gpu": point,
                "summary": detailed["summary"],
            }
            evaluated.append(item)
        except (Llm.Error, AssertionError, ValueError) as exc:
            logger.debug("GPU design point rejected: %s", exc)
    if not evaluated:
        return {
            "status": "error",
            "error": "No acceptable GPU design around the selected model",
            "baseline_gpu": baseline,
        }, []
    ranked = _rank_metric_items(evaluated, tuple(objectives) + ("gpu_cost",))
    for index, item in enumerate(ranked, start=1):
        public_candidates.append({
            "candidate_id": index,
            "gpu": item["gpu"],
            "metrics": item["metrics"],
            "balanced_score": item["balanced_score"],
            "pareto_optimal": item["pareto_optimal"],
        })
    best = ranked[0]
    baseline_item = min(
        evaluated,
        key=lambda item: (
            abs(item["gpu"]["compute_factor"] - 1.0)
            + abs(item["gpu"]["memory_capacity_factor"] - 1.0)
            + abs(item["gpu"]["memory_bandwidth_factor"] - 1.0)),
    )
    baseline_throughput = float(
        baseline_item["metrics"]["linear_scaling_throughput"])
    best_throughput = float(best["metrics"]["linear_scaling_throughput"])
    conclusion = {
        "status": "ok",
        "baseline_gpu": baseline,
        "optimal_gpu": best["gpu"],
        "fixed_hardware": {
            key: hardware.get(key) for key in HARDWARE_IDENTITY_FIELDS
        },
        "fixed_software": parallel,
        "objectives": list(objectives),
        "metrics": best["metrics"],
        "balanced_score": best["balanced_score"],
        "pareto_optimal": best["pareto_optimal"],
        "throughput_change_vs_baseline_percent": (
            (best_throughput / baseline_throughput - 1.0) * 100.0
            if baseline_throughput > 0 else None),
        "selection_method": (
            "Pareto minimax/mean normalized performance loss plus normalized "
            "GPU resource investment"),
        "evaluated_gpu_designs": len(evaluated),
    }
    return conclusion, public_candidates

def _deduplicate_software(candidates: Sequence[Dict], limit: int) -> List[Dict]:
    selected = []
    seen = set()
    for candidate in candidates:
        identity = tuple(sorted(candidate.items()))
        if identity in seen:
            continue
        seen.add(identity)
        selected.append(candidate)
        if len(selected) >= limit:
            break
    return selected


def _matches_public_software(candidate: Dict, public: Dict) -> bool:
    fields = {
        "tp": "tensor_parallel",
        "pp": "pipeline_parallel",
        "dp": "data_parallel",
        "ep": "expert_parallel",
        "cp": "context_parallel",
        "batch_size": "batch_size",
        "microbatch_size": "microbatch_size",
        "pipeline_interleaving": "pipeline_interleaving",
        "activation_recompute": "activation_recompute",
        "optimizer_sharding": "optimizer_sharding",
        "tensor_par_comm_type": "tensor_parallel_comm_type",
        "tensor_par_overlap": "tensor_parallel_overlap",
        "weight_offload": "weight_offload",
        "activations_offload": "activations_offload",
        "optimizer_offload": "optimizer_offload",
        "data_par_overlap": "data_parallel_overlap",
    }
    return all(candidate.get(source) == public.get(target)
               for source, target in fields.items())


CHART_HARDWARE_AXES = (
    "scale_up_size", "intra_bandwidth", "inter_pod_bandwidth",
    "intra_latency", "inter_latency",
)


def _candidate_record_matches_software(candidate: Dict,
                                       public: Dict,
                                       policy: str,
                                       comparison_batch: int) -> bool:
    parallel = candidate.get("parallelism") or {}
    software_fields = (
        "tensor_parallel", "pipeline_parallel", "data_parallel",
        "expert_parallel", "context_parallel", "batch_size",
        "microbatch_size", "pipeline_interleaving", "activation_recompute",
        "optimizer_sharding", "tensor_parallel_comm_type",
        "tensor_parallel_overlap", "data_parallel_overlap",
        "weight_offload", "activations_offload", "optimizer_offload",
    )
    return (
        int(parallel.get("batch_size") or 0) == comparison_batch
        and all(parallel.get(field) == public.get(field)
                for field in software_fields)
        and (candidate.get("placement") or {}).get("policy") == policy
    )


def _ranked_chart_hardware_contexts(candidate_records: Sequence[Dict],
                                    optimal_hardware: Dict,
                                    public_software: Dict,
                                    policy: str,
                                    comparison_batch: int,
                                    objectives: Sequence[str]) -> List[Dict]:
    """Rank hardware anchors after fixing software, workload, and placement."""
    matching = [
        candidate for candidate in candidate_records
        if _candidate_record_matches_software(
            candidate, public_software, policy, comparison_batch)
    ]
    ranked = _rank_metric_items(matching, objectives) if matching else []
    contexts = [dict(optimal_hardware)]
    seen = {_hardware_identity(optimal_hardware)}
    for candidate in ranked:
        hardware = candidate.get("hardware") or {}
        identity = _hardware_identity(hardware)
        if identity in seen:
            continue
        seen.add(identity)
        contexts.append(dict(hardware))
    return contexts


def _controlled_chart_points(coarse_points: Sequence[Dict],
                             ranked_contexts: Sequence[Dict],
                             contexts_per_axis: int) -> List[Dict]:
    """Build fair X sweeps for several strong, fixed hardware contexts."""
    points = []
    seen_points = set()
    for axis in CHART_HARDWARE_AXES:
        values = sorted({point.get(axis) for point in coarse_points
                         if point.get(axis) is not None})
        selected_contexts = []
        seen_contexts = set()
        for hardware in ranked_contexts:
            context_identity = tuple(
                hardware.get(key) for key in CHART_HARDWARE_AXES
                if key != axis
            ) + (hardware.get("network_topology"),)
            if context_identity in seen_contexts:
                continue
            seen_contexts.add(context_identity)
            selected_contexts.append(hardware)
            if len(selected_contexts) >= contexts_per_axis:
                break

        for context_rank, hardware in enumerate(selected_contexts, start=1):
            fixed_num_procs = int(hardware["num_procs"])
            axis_values = sorted(set(values) | {hardware.get(axis)})
            for value in axis_values:
                point = dict(hardware)
                point[axis] = value
                scale_up = int(point["scale_up_size"])
                if scale_up <= 0 or fixed_num_procs % scale_up:
                    continue
                point["scale_out_size"] = fixed_num_procs // scale_up
                point["inter_pod_bandwidth"] = float(
                    point["inter_pod_bandwidth"])
                point["inter_bandwidth"] = (
                    point["inter_pod_bandwidth"] / scale_up)
                point_identity = (
                    axis,
                    tuple(point.get(key) for key in CHART_HARDWARE_AXES),
                    point.get("network_topology"),
                )
                if point_identity in seen_points:
                    continue
                seen_points.add(point_identity)
                point["sweep_dimension"] = axis
                point["chart_controlled"] = True
                point["chart_context_rank"] = context_rank
                point["chart_series_context"] = {
                    key: point.get(key) for key in CHART_HARDWARE_AXES
                    if key != axis
                }
                point["chart_series_context"]["network_topology"] = (
                    point.get("network_topology"))
                points.append(point)
    return points


def _fixed_software_chart(logger, app: Llm.Application, config,
                          base_system_json: Dict, points: Sequence[Dict],
                          software_candidate: Dict, policy: str,
                          profile_cache: Dict) -> List[Dict]:
    records = []
    chart_config = config.model_copy(update={
        "max_candidates": 1, "coarse_top_k": 1, "top_n": 1,
    })
    for hardware in points:
        def factory(placement, point=hardware):
            return build_candidate_system(base_system_json, point, placement,
                                          logger)
        result = ConstrainedSearch.run(
            logger, app, chart_config, hardware, factory, (policy,),
            software_candidates=([software_candidate], 1),
            profile_cache=profile_cache)
        if result.get("status") != "error":
            records.extend(result.get("search_candidates", []))
    return records



def run_hardware_design(logger, app: Llm.Application, base_system_json: Dict,
                        config, defaults: Dict,
                        gpu_name: Optional[str] = None) -> Dict:
    objectives, training_samples = validate_objective(config)
    total_hw = successful_hw = total_executions = duplicate_placements = 0
    cache_hits = cache_misses = 0
    all_candidates: List[Dict] = []
    evaluated_designs: List[Dict] = []
    coarse_points = list(hardware_points(config, defaults))
    policies = tuple(getattr(config, "placement_policies", None)
                     or (DEFAULT_MOE_POLICIES if app.is_moe else DEFAULT_DENSE_POLICIES))
    policy_count = max(1, len(_unique(policies)))
    requested_budget = max(1, int(getattr(config, "max_candidates", None) or 512))
    refinement_limit = max(
        0, int(getattr(config, "adaptive_refinement_points", None) or 0))
    software_top_k = max(
        1, int(getattr(config, "software_top_k", None) or 4))
    software_screen_limit = min(
        requested_budget,
        max(software_top_k,
            int(getattr(config, "coarse_top_k", None) or 128)))
    profile_cache: Dict[Tuple, Dict] = {}
    if not coarse_points:
        return {"status": "error", "error": "No legal hardware points found"}

    baseline_hardware = next(
        (point for point in coarse_points
         if point.get("sweep_dimension") == "baseline"),
        coarse_points[0],
    )
    canonical_scale_up = min(
        int(config.num_procs),
        int(getattr(config, "max_scale_up_size", None) or config.num_procs),
    )
    screen_config = config.model_copy(update={
        "max_candidates": software_screen_limit,
        "coarse_top_k": software_screen_limit,
        "top_n": software_top_k,
    })
    generated_software = expand_software_candidates(
        app, screen_config, canonical_scale_up)

    def screen_factory(placement):
        return build_candidate_system(
            base_system_json, baseline_hardware, placement, logger)

    logger.info(
        "Superpod stage 1/2 software screening: baseline=%s candidates=%d "
        "top_k=%d policies=%d",
        _hardware_identity(baseline_hardware), len(generated_software[0]),
        software_top_k, policy_count)
    screen_result = ConstrainedSearch.run(
        logger, app, screen_config, baseline_hardware, screen_factory, policies,
        software_candidates=generated_software, profile_cache=profile_cache)
    screen_stats = screen_result.get("executions", {})
    total_executions += screen_stats.get("total_executions", 0)
    duplicate_placements += screen_stats.get(
        "duplicate_placements_pruned", 0)
    cache_hits += screen_stats.get("profile_cache_hits", 0)
    cache_misses += screen_stats.get("profile_cache_misses", 0)
    selected_software = _deduplicate_software(
        screen_result.get("_ranked_software_candidates", []), software_top_k)
    if not selected_software:
        return {
            "status": "error",
            "error": "No acceptable software strategy found at baseline hardware",
            "error_summary": screen_result.get("error_summary", {}),
        }
    software_candidates = (selected_software, generated_software[1])
    software_per_hardware = len(selected_software)
    search_config = config.model_copy(update={
        "max_candidates": software_per_hardware * policy_count,
        "coarse_top_k": software_per_hardware * policy_count,
    })

    logger.info(
        "Superpod four-stage plan: coarse_points=%d refinement_limit=%d "
        "policies=%d budget=%d software_per_hardware=%d",
        len(coarse_points), refinement_limit, policy_count, requested_budget,
        software_per_hardware)

    def evaluate(points: Sequence[Dict], phase: str) -> None:
        nonlocal total_hw, successful_hw, total_executions
        nonlocal duplicate_placements, cache_hits, cache_misses
        for hardware in points:
            total_hw += 1
            point_config = search_config.model_copy(
                update={"num_procs": int(hardware["num_procs"])})

            def factory(placement, point=hardware):
                return build_candidate_system(base_system_json, point, placement, logger)

            result = ConstrainedSearch.run(
                logger, app, point_config, hardware, factory, policies,
                software_candidates=software_candidates,
                profile_cache=profile_cache)
            result.pop("_ranked_software_candidates", None)
            execution_stats = result.get("executions", {})
            total_executions += execution_stats.get("total_executions", 0)
            duplicate_placements += execution_stats.get(
                "duplicate_placements_pruned", 0)
            cache_hits += execution_stats.get("profile_cache_hits", 0)
            cache_misses += execution_stats.get("profile_cache_misses", 0)
            all_candidates.extend(result.get("search_candidates", []))
            if result.get("status") == "error":
                continue
            successful_hw += 1
            merged_hardware = {**hardware, **result.get("hardware", {})}
            result["hardware"] = merged_hardware
            result["search_phase"] = phase
            evaluated_designs.append({
                "metrics": _summary_metrics(result["summary"]),
                "result": result,
            })

    # Stage 1/2 are performed by candidate generation and cheap lower-bound
    # ordering in expand_software_candidates; Stage 3 begins with coarse points.
    evaluate(coarse_points, "coarse")
    refined_points = _adaptive_hardware_points(
        coarse_points, evaluated_designs, refinement_limit)
    evaluate(refined_points, "adaptive")

    ranked, comparison_batch, fairness = _rank_hardware_fair(
        evaluated_designs, all_candidates, objectives)
    top = max(1, int(getattr(config, "hardware_top_n", None) or 10))
    selected = ranked[:top]
    if not selected:
        return {"status": "error", "error": "No acceptable hardware/software design found",
                "hardware_executions": {"hardware_points": total_hw,
                                        "total_executions": total_executions}}

    selected_hardware = selected[0]["result"]["hardware"]
    best = dict(selected[0]["result"])
    # Re-run only the pre-screened software strategies at the common batch.
    fixed_batch_software = [
        candidate for candidate in selected_software
        if int(candidate["batch_size"]) == comparison_batch
    ]
    if comparison_batch > 0 and fixed_batch_software:
        final_candidate_limit = len(fixed_batch_software) * policy_count
        final_config = config.model_copy(update={
            "num_procs": int(selected_hardware["num_procs"]),
            "max_global_batch_size": None,
            "global_batch_size": comparison_batch,
            "max_candidates": final_candidate_limit,
            "coarse_top_k": final_candidate_limit,
        })

        def final_factory(placement):
            return build_candidate_system(
                base_system_json, selected_hardware, placement, logger)

        final_result = ConstrainedSearch.run(
            logger, app, final_config, selected_hardware, final_factory,
            policies,
            software_candidates=(fixed_batch_software, generated_software[1]),
            profile_cache=profile_cache)
        final_stats = final_result.get("executions", {})
        total_executions += final_stats.get("total_executions", 0)
        if final_result.get("status") != "error":
            final_result["hardware"] = {
                **selected_hardware, **final_result.get("hardware", {})}
            final_result.pop("_ranked_software_candidates", None)
            best = final_result
    best["summary"] = dict(best["summary"])
    placement = best.get("placement", {})
    groups = placement.get("groups", {})
    mapping = {}
    for dimension, group in groups.items():
        local = float(group.get("local_fraction", 0.0))
        remote = float(group.get("remote_fraction", 0.0))
        mapping[dimension.upper()] = (
            "Scale-up" if remote <= 0 else
            "Scale-out" if local <= 0 else
            f"Mixed ({local:.1%} scale-up / {remote:.1%} scale-out)")
    recommended = {
        **best["hardware"],
        "placement_policy": placement.get("policy"),
        "parallel_dimension_mapping": mapping,
    }
    knees = _bandwidth_latency_knees(ranked)
    best["summary"]["recommended_hardware"] = recommended
    best["summary"]["knees_95_percent"] = knees
    best["summary"]["objectives"] = list(objectives)
    best["summary"]["pareto_optimal"] = selected[0]["pareto_optimal"]
    best["summary"]["balanced_score"] = selected[0]["balanced_score"]
    best["summary"]["comparison_batch_size"] = comparison_batch
    best["summary"]["fair_comparison"] = fairness
    best["recommended_hardware"] = recommended
    best["knees_95_percent"] = knees

    gpu_conclusion, gpu_candidates = _search_gpu_design(
        logger, app, base_system_json, config, best, objectives,
        training_samples)
    hardware_conclusion = {
        "title": "Conclusion 1: Optimal Superpod Design with Fixed GPU Model",
        "fixed_gpu": {
            "name": gpu_name,
            **gpu_conclusion.get("baseline_gpu", {}),
        },
        "optimal_hardware": recommended,
        "optimal_software": best["optimal_result"],
        "objectives": list(objectives),
        "metrics": _summary_metrics(best["summary"]),
        "comparison_batch_size": comparison_batch,
        "balanced_score": selected[0]["balanced_score"],
        "pareto_optimal": selected[0]["pareto_optimal"],
        "fair_comparison": fairness,
        "selection_method": (
            "One common global batch; metrics normalized within identical "
            "software/placement signatures; global Pareto robust regret "
            "balances worst case, mean case, and coverage; hardware investment only breaks ties"),
    }
    gpu_conclusion["title"] = (
        "Conclusion 2: Optimal GPU Design with Fixed Superpod and Software")
    conclusions = {
        "hardware_design": hardware_conclusion,
        "gpu_design": gpu_conclusion,
    }
    best["summary"]["design_conclusions"] = conclusions
    best["design_conclusions"] = conclusions
    best["gpu_design_candidates"] = gpu_candidates

    optimal_public_software = best.get("optimal_result", {})
    chart_software = next(
        (candidate for candidate in selected_software
         if _matches_public_software(candidate, optimal_public_software)),
        selected_software[0],
    )
    chart_policy = (best.get("placement") or {}).get("policy") or policies[0]
    ranked_chart_contexts = _ranked_chart_hardware_contexts(
        all_candidates, selected_hardware, optimal_public_software,
        chart_policy, comparison_batch, objectives)
    chart_context_limit = max(
        1, min(3, int(getattr(config, "hardware_top_n", None) or 10)))
    chart_points = _controlled_chart_points(
        coarse_points, ranked_chart_contexts, chart_context_limit)
    comparable_candidates = _fixed_software_chart(
        logger, app, config, base_system_json, chart_points,
        chart_software, chart_policy, profile_cache)
    for candidate_id, candidate in enumerate(comparable_candidates, start=1):
        candidate["candidate_id"] = candidate_id
        candidate["comparison_batch_size"] = comparison_batch
    best["search_candidates"] = comparable_candidates
    best["search_candidate_context"] = {
        "fixed_software": optimal_public_software,
        "fixed_placement_policy": chart_policy,
        "software_shortlist_size": software_per_hardware,
        "hardware_contexts_per_axis": chart_context_limit,
        "hardware_comparison": (
            "Software, workload, and placement are fixed to the final optimal "
            "strategy. Each legend series fixes every hardware parameter "
            "except X; its fixed values come from a globally ranked hardware "
            "context."),
        "controlled_axes": list(CHART_HARDWARE_AXES),
    }
    best["hardware_results"] = [item["result"] for item in selected]
    effective_budget = software_per_hardware * total_hw * policy_count
    best["hardware_executions"] = {
        "hardware_points": total_hw,
        "coarse_hardware_points": len(coarse_points),
        "adaptive_hardware_points": len(refined_points),
        "successful_hardware_points": successful_hw,
        "total_executions": total_executions,
        "duplicate_placements_pruned": duplicate_placements,
        "requested_execution_budget": requested_budget,
        "effective_execution_budget": effective_budget,
        "software_candidates_per_hardware": software_per_hardware,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
    }
    best["search_phases"] = {
        "stage1": (
            "Generate legal 5D/recompute/sharding/overlap/offload strategies "
            "and reject invalid or OOM candidates at baseline hardware"),
        "stage2": {
            "method": "exact baseline ranking after cheap lower-bound ordering",
            "screened": software_screen_limit,
            "selected_software_top_k": software_per_hardware,
        },
        "stage3": {
            "method": (
                "search hardware using only the fixed software shortlist; "
                "then generate controlled one-axis chart sweeps"),
            "coarse_points": len(coarse_points),
            "adaptive_points": len(refined_points),
            "controlled_chart_points": len(comparable_candidates),
        },
        "stage4": {
            "cache_key": "(model, dtype, 5D, microbatch, recompute, sharding, overlap, offload)",
            "cached_values": "operator compute, memory, and collective communication volumes",
            "software_candidate_cache_entries": 1,
            "static_software_profile_cache_entries": len(profile_cache),
        },
    }
    best["modeling_notes"] = [
        "Placement uses contiguous hierarchical rank order.",
        "Mixed local/remote traffic uses harmonic effective bandwidth.",
        "Inter-node bandwidth is aggregate per scale-up pod and is converted to per-GPU injection bandwidth.",
    ]
    return best


def _placement_signature(placement: PlacementResult) -> Tuple:
    """Return the simulator-relevant placement shape, ignoring policy labels."""
    return (
        tuple((dimension,
               group.network_tier,
               round(group.local_fraction, 12),
               round(group.remote_fraction, 12),
               round(group.effective_bandwidth, 6),
               round(group.effective_latency, 12))
              for dimension, group in sorted(placement.groups.items())),
    )

