import numpy as np
from dataclasses import dataclass
from itertools import permutations
from typing import Dict, List, Optional, Tuple
from scipy.optimize import linear_sum_assignment
import os
import sys
import base64
import pika
import pickle
import src.Model
import src.Log
from ultralytics import YOLO


DEVICE_A_4CORE = np.array([
    0.001433, 0.00342, 0.004968, 0.006701, 0.004795, 0.006678,
    0.003975, 0.003339, 0.003928, 0.001866, 0.00285, 0.000012,
    0.0, 0.005454, 0.000023, 0.0, 0.006286, 0.001675, 0.0,
    0.004344, 0.00167, 0.0, 0.005262, 0.017629
], dtype=float)

DEVICE_B_2CORE = np.array([
    0.002286, 0.005456, 0.007927, 0.010692, 0.00765, 0.010655,
    0.006341, 0.005327, 0.006268, 0.002977, 0.004546, 0.000018,
    0.0, 0.008701, 0.000037, 0.0, 0.010028, 0.002673, 0.0,
    0.006931, 0.002664, 0.0, 0.008394, 0.028126
], dtype=float)

DEVICE_C_1CORE = np.array([
    0.00327, 0.007807, 0.011341, 0.015297, 0.010946, 0.015245,
    0.009073, 0.007622, 0.008967, 0.00426, 0.006505, 0.000026,
    0.0, 0.012449, 0.000053, 0.0, 0.014348, 0.003824, 0.0,
    0.009917, 0.003811, 0.0, 0.01201, 0.040242
], dtype=float)

DEVICE_CLOUD = np.array([
    0.000534, 0.001272, 0.001851, 0.002496, 0.001785, 0.002487,
    0.001479, 0.001242, 0.001461, 0.000696, 0.001062, 0.000003,
    0.0, 0.002031, 0.000009, 0.0, 0.00234, 0.000624, 0.0,
    0.001617, 0.000621, 0.0, 0.001959, 0.006564
], dtype=float)

# yolo11n timing profiles (profiled then scaled by same HW ratios as yolo26n)
DEVICE_A_4CORE_yolo11n = np.array([
    0.006445, 0.006742, 0.018843, 0.007665, 0.010089, 0.006162,
    0.007317, 0.002917, 0.006075, 0.010867, 0.006786, 0.001811,
    0.000546, 0.007257, 0.001205, 0.001856, 0.011045, 0.001954,
    0.000292, 0.005754, 0.001749, 0.000152, 0.006325, 0.040976
], dtype=float)

DEVICE_B_2CORE_yolo11n = DEVICE_A_4CORE_yolo11n * 1.595
DEVICE_C_1CORE_yolo11n = DEVICE_A_4CORE_yolo11n * 2.282
DEVICE_CLOUD_yolo11n   = DEVICE_A_4CORE_yolo11n * 0.373

def _select_timing_profiles(model_name: str):
    if "11" in model_name:
        return (DEVICE_A_4CORE_yolo11n, DEVICE_B_2CORE_yolo11n,
                DEVICE_C_1CORE_yolo11n, DEVICE_CLOUD_yolo11n)
    return DEVICE_A_4CORE, DEVICE_B_2CORE, DEVICE_C_1CORE, DEVICE_CLOUD

CUT_DATA_SIZES_MB = np.array([
    13.78, 8.02, 20.5, 5.81, 9.61, 12.54, 12.38, 13.61, 13.71,
    13.83, 13.52, 18.11, 17.89, 13.83, 26.1, 23.93, 9.94, 11.29,
    10.87, 9.57, 10.27, 10.25, 9.42
], dtype=float)

RAW_INPUT_MB = 13.0


@dataclass
class ScenarioConfig:
    num_A: int
    num_B: int
    num_C: int
    num_cloud: int
    network_rate_mb_s: float
    network_overhead_s: float
    input_data_mb: float = RAW_INPUT_MB
    edge_time_jitter: float = 0.0
    cloud_time_jitter: float = 0.0


@dataclass
class PairDetail:
    edge_cluster_id: int
    cloud_cluster_id: int
    clients: List[int]
    servers: List[int]
    best_cut: int
    producer_rate: float
    service_rate: float
    throughput: float
    round_time: float


@dataclass
class OptimizationResult:
    method: str
    total_throughput: float
    system_round_time_proxy: float
    num_clusters: int
    edge_labels: np.ndarray
    cloud_labels: np.ndarray
    matching: np.ndarray
    best_cuts: np.ndarray
    details: List[PairDetail]


@dataclass
class ManualExperimentConfig:
    num_A: int = 3
    num_B: int = 3
    num_C: int = 3
    num_cloud: int = 3

    network_rate_mb_s: float = 100.0
    network_overhead_s: float = 0.0

    network_rates_matrix: Optional[np.ndarray] = None
    network_overheads_matrix: Optional[np.ndarray] = None

    input_data_mb: float = RAW_INPUT_MB
    edge_time_jitter: float = 0.0
    cloud_time_jitter: float = 0.0

    max_clusters: Optional[int] = None
    cluster_penalty: float = 0.0
    exact_max_k: int = 7


class ScenarioSampler:
    def __init__(
        self,
        min_A: int,
        max_A: int,
        min_B: int,
        max_B: int,
        min_C: int,
        max_C: int,
        min_cloud: int,
        max_cloud: int,
        network_rate_range: Tuple[float, float],
        network_overhead_range: Tuple[float, float] = (0.0, 0.0),
        edge_time_jitter_range: Tuple[float, float] = (0.0, 0.0),
        cloud_time_jitter_range: Tuple[float, float] = (0.0, 0.0),
        input_data_mb: float = RAW_INPUT_MB,
        seed: int = 0,
    ):
        self.min_A = min_A
        self.max_A = max_A
        self.min_B = min_B
        self.max_B = max_B
        self.min_C = min_C
        self.max_C = max_C
        self.min_cloud = min_cloud
        self.max_cloud = max_cloud
        self.network_rate_range = network_rate_range
        self.network_overhead_range = network_overhead_range
        self.edge_time_jitter_range = edge_time_jitter_range
        self.cloud_time_jitter_range = cloud_time_jitter_range
        self.input_data_mb = float(input_data_mb)
        self.rng = np.random.default_rng(seed)

    def sample(self):
        return ScenarioConfig(
            num_A=int(self.rng.integers(self.min_A, self.max_A + 1)),
            num_B=int(self.rng.integers(self.min_B, self.max_B + 1)),
            num_C=int(self.rng.integers(self.min_C, self.max_C + 1)),
            num_cloud=int(self.rng.integers(self.min_cloud, self.max_cloud + 1)),
            network_rate_mb_s=float(self.rng.uniform(*self.network_rate_range)),
            network_overhead_s=float(self.rng.uniform(*self.network_overhead_range)),
            input_data_mb=self.input_data_mb,
            edge_time_jitter=float(self.rng.uniform(*self.edge_time_jitter_range)),
            cloud_time_jitter=float(self.rng.uniform(*self.cloud_time_jitter_range)),
        )


def make_default_samplers():
    train = ScenarioSampler(
        2, 12, 2, 12, 2, 12, 1, 6,
        (40.0, 800.0), (0.0, 0.05), (0.0, 0.18), (0.0, 0.10),
        seed=42
    )
    val = ScenarioSampler(
        2, 12, 2, 12, 2, 12, 1, 6,
        (30.0, 900.0), (0.0, 0.06), (0.0, 0.20), (0.0, 0.12),
        seed=123
    )
    test = ScenarioSampler(
        2, 15, 2, 15, 2, 15, 1, 8,
        (20.0, 1000.0), (0.0, 0.08), (0.0, 0.25), (0.0, 0.15),
        seed=999
    )
    return train, val, test


def _jitter_profile(base: np.ndarray, jitter: float, rng: np.random.Generator):
    if jitter <= 0:
        return base.copy()
    noise = rng.uniform(1.0 - jitter, 1.0 + jitter, size=base.shape)
    out = base * noise
    out[base == 0.0] = 0.0
    return out


def build_scenario_from_config(config: ScenarioConfig, rng: np.random.Generator):
    client_blocks = []
    client_type_names = []

    for _ in range(config.num_A):
        client_blocks.append(_jitter_profile(DEVICE_A_4CORE, config.edge_time_jitter, rng))
        client_type_names.append("A")
    for _ in range(config.num_B):
        client_blocks.append(_jitter_profile(DEVICE_B_2CORE, config.edge_time_jitter, rng))
        client_type_names.append("B")
    for _ in range(config.num_C):
        client_blocks.append(_jitter_profile(DEVICE_C_1CORE, config.edge_time_jitter, rng))
        client_type_names.append("C")

    if len(client_blocks) == 0:
        raise ValueError("At least one edge device is required.")
    if config.num_cloud <= 0:
        raise ValueError("At least one cloud server is required.")

    client_layer_times = np.vstack(client_blocks)
    server_layer_times = np.vstack([
        _jitter_profile(DEVICE_CLOUD, config.cloud_time_jitter, rng)
        for _ in range(config.num_cloud)
    ])

    N = client_layer_times.shape[0]
    M = server_layer_times.shape[0]
    network_rates = np.full((N, M), config.network_rate_mb_s, dtype=float)
    network_overheads = np.full((N, M), config.network_overhead_s, dtype=float)

    solver = DeterministicSimilarityAssignmentSolver(
        client_layer_times=client_layer_times,
        server_layer_times=server_layer_times,
        cut_data_sizes=CUT_DATA_SIZES_MB,
        input_data_size=config.input_data_mb,
        network_rates=network_rates,
        network_overheads=network_overheads,
    )
    solver.client_type_names = client_type_names
    return solver, config


class DeterministicSimilarityAssignmentSolver:
    def __init__(
        self,
        client_layer_times: np.ndarray,
        server_layer_times: np.ndarray,
        cut_data_sizes: np.ndarray,
        input_data_size: float,
        network_rates: np.ndarray,
        network_overheads: Optional[np.ndarray] = None,
        eps: float = 1e-9,
    ):
        self.client_layer_times = np.asarray(client_layer_times, dtype=float)
        self.server_layer_times = np.asarray(server_layer_times, dtype=float)
        self.cut_data_sizes = np.asarray(cut_data_sizes, dtype=float)
        self.input_data_size = float(input_data_size)
        self.network_rates = np.asarray(network_rates, dtype=float)
        self.eps = float(eps)

        self.N, self.L = self.client_layer_times.shape
        self.M, L2 = self.server_layer_times.shape
        if L2 != self.L:
            raise ValueError("client_layer_times and server_layer_times must have same number of layers")
        if self.cut_data_sizes.shape != (self.L - 1,):
            raise ValueError(f"cut_data_sizes must have shape ({self.L - 1},)")
        if self.network_rates.shape != (self.N, self.M):
            raise ValueError(f"network_rates must have shape ({self.N}, {self.M})")
        if np.any(self.network_rates <= 0):
            raise ValueError("All network_rates must be > 0")

        if network_overheads is None:
            self.network_overheads = np.zeros((self.N, self.M), dtype=float)
        else:
            self.network_overheads = np.asarray(network_overheads, dtype=float)
            if self.network_overheads.shape != (self.N, self.M):
                raise ValueError(f"network_overheads must have shape ({self.N}, {self.M})")

        self.valid_cuts = list(range(-1, self.L))
        self.client_prefix = np.cumsum(self.client_layer_times, axis=1)
        self.server_suffix = np.flip(np.cumsum(np.flip(self.server_layer_times, axis=1), axis=1), axis=1)
        self.client_total = self.client_prefix[:, -1]
        self.server_total = np.sum(self.server_layer_times, axis=1)
        self.cluster_cache: Dict[int, Dict[str, object]] = {}
        self.pair_cache: Dict[int, Dict[Tuple[int, int], PairDetail]] = {}

    def edge_time(self, client_id: int, cut: int):
        if cut < 0:
            return 0.0
        if cut >= self.L - 1:
            return float(self.client_total[client_id])
        return float(self.client_prefix[client_id, cut])

    def cloud_time(self, server_id: int, cut: int):
        if cut < 0:
            return float(self.server_total[server_id])
        if cut >= self.L - 1:
            return 0.0
        return float(self.server_suffix[server_id, cut + 1])

    def net_time(self, client_id: int, server_id: int, cut: int):
        if cut < 0:
            return self.input_data_size / self.network_rates[client_id, server_id] + self.network_overheads[client_id, server_id]
        if cut >= self.L - 1:
            return 0.0
        return self.cut_data_sizes[cut] / self.network_rates[client_id, server_id] + self.network_overheads[client_id, server_id]

    def _normalize_features(self, x: np.ndarray):
        mins = x.min(axis=0, keepdims=True)
        maxs = x.max(axis=0, keepdims=True)
        denom = np.maximum(maxs - mins, self.eps)
        return (x - mins) / denom

    def build_edge_features(self):
        mean_rate = np.mean(self.network_rates, axis=1, keepdims=True)
        mean_ovh = np.mean(self.network_overheads, axis=1, keepdims=True)
        raw = np.concatenate([self.client_layer_times, mean_rate, mean_ovh], axis=1)
        return self._normalize_features(raw)

    def build_cloud_features(self):
        mean_rate = np.mean(self.network_rates, axis=0, keepdims=True).T
        mean_ovh = np.mean(self.network_overheads, axis=0, keepdims=True).T
        raw = np.concatenate([self.server_layer_times, mean_rate, mean_ovh], axis=1)
        return self._normalize_features(raw)

    def agglomerative_cluster(self, features: np.ndarray, K: int):
        n = features.shape[0]
        if K <= 0 or K > n:
            raise ValueError("Invalid K for clustering")
        if K == n:
            return np.arange(n, dtype=int)
        if K == 1:
            return np.zeros(n, dtype=int)

        diff = features[:, None, :] - features[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        clusters = [[i] for i in range(n)]

        def avg_linkage(c1, c2):
            vals = [dist[a, b] for a in c1 for b in c2]
            return float(np.mean(vals))

        while len(clusters) > K:
            best_pair = None
            best_val = float("inf")
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    val = avg_linkage(clusters[i], clusters[j])
                    if val < best_val:
                        best_val = val
                        best_pair = (i, j)
            i, j = best_pair
            merged = clusters[i] + clusters[j]
            new_clusters = []
            for idx, c in enumerate(clusters):
                if idx not in (i, j):
                    new_clusters.append(c)
            new_clusters.append(sorted(merged))
            clusters = new_clusters

        clusters = sorted(clusters, key=lambda c: min(c))
        labels = np.zeros(n, dtype=int)
        for k, members in enumerate(clusters):
            for idx in members:
                labels[idx] = k
        return labels

    def _invert_assignment(self, assign: np.ndarray, K: int):
        groups = {k: [] for k in range(K)}
        for idx, g in enumerate(assign):
            groups[int(g)].append(idx)
        return groups

    def build_similarity_clusters(self, K: int):
        if K in self.cluster_cache:
            return self.cluster_cache[K]
        edge_labels = self.agglomerative_cluster(self.build_edge_features(), K)
        cloud_labels = self.agglomerative_cluster(self.build_cloud_features(), K)
        out = {
            "edge_labels": edge_labels,
            "cloud_labels": cloud_labels,
            "edge_groups": self._invert_assignment(edge_labels, K),
            "cloud_groups": self._invert_assignment(cloud_labels, K),
        }
        self.cluster_cache[K] = out
        return out

    def pair_metrics_for_cut(self, edge_cluster_id: int, cloud_cluster_id: int, clients: List[int], servers: List[int], cut: int):
        client_prod_time = {}
        for i in clients:
            best_ingress = min(self.net_time(i, j, cut) for j in servers)
            tau_prod = max(self.edge_time(i, cut) + best_ingress, self.eps)
            client_prod_time[i] = float(tau_prod)

        server_service_time = {}
        for j in servers:
            server_service_time[j] = float(self.cloud_time(j, cut))

        producer_rate = float(np.sum([1.0 / t for t in client_prod_time.values()]))
        if all(t <= self.eps for t in server_service_time.values()):
            service_rate = float("inf")
        else:
            service_rate = float(np.sum([1.0 / max(t, self.eps) for t in server_service_time.values()]))

        throughput = min(producer_rate, service_rate)
        round_time = len(clients) / max(throughput, self.eps)

        return PairDetail(
            edge_cluster_id=edge_cluster_id,
            cloud_cluster_id=cloud_cluster_id,
            clients=clients,
            servers=servers,
            best_cut=cut,
            producer_rate=producer_rate,
            service_rate=service_rate,
            throughput=throughput,
            round_time=round_time,
        )

    def best_cut_for_pair(self, edge_cluster_id: int, cloud_cluster_id: int, clients: List[int], servers: List[int]):
        best_detail = None
        best_throughput = -float("inf")
        best_round_time = float("inf")
        for cut in self.valid_cuts:
            detail = self.pair_metrics_for_cut(edge_cluster_id, cloud_cluster_id, clients, servers, cut)
            if (detail.throughput > best_throughput) or (
                np.isclose(detail.throughput, best_throughput) and detail.round_time < best_round_time
            ):
                best_throughput = detail.throughput
                best_round_time = detail.round_time
                best_detail = detail
        return best_detail

    def build_pair_cache(self, K: int):
        if K in self.pair_cache:
            return self.pair_cache[K]
        cl = self.build_similarity_clusters(K)
        edge_groups = cl["edge_groups"]
        cloud_groups = cl["cloud_groups"]
        out = {}
        for e in range(K):
            for c in range(K):
                out[(e, c)] = self.best_cut_for_pair(e, c, edge_groups[e], cloud_groups[c])
        self.pair_cache[K] = out
        return out

    def get_weight_matrix(self, K: int):
        cl = self.build_similarity_clusters(K)
        pair_cache = self.build_pair_cache(K)
        W = np.zeros((K, K), dtype=float)
        for e in range(K):
            for c in range(K):
                W[e, c] = pair_cache[(e, c)].throughput
        return W, cl, pair_cache

    def _result_from_matching(self, K: int, matching: np.ndarray, method: str):
        cl = self.build_similarity_clusters(K)
        pair_cache = self.build_pair_cache(K)
        total_throughput = 0.0
        details = []
        best_cuts = []

        for e in range(K):
            c = int(matching[e])
            detail = pair_cache[(e, c)]
            total_throughput += detail.throughput
            details.append(detail)
            best_cuts.append(detail.best_cut)

        system_round_time_proxy = max(d.round_time for d in details)

        return OptimizationResult(
            method=method,
            total_throughput=float(total_throughput),
            system_round_time_proxy=float(system_round_time_proxy),
            num_clusters=K,
            edge_labels=cl["edge_labels"].copy(),
            cloud_labels=cl["cloud_labels"].copy(),
            matching=matching.copy(),
            best_cuts=np.array(best_cuts, dtype=int),
            details=details,
        )

    def solve_identity_for_k(self, K: int):
        return self._result_from_matching(K, np.arange(K, dtype=int), "identity")

    def solve_greedy_for_k(self, K: int):
        pair_cache = self.build_pair_cache(K)
        remaining_e = set(range(K))
        remaining_c = set(range(K))
        matching = np.full(K, -1, dtype=int)

        while remaining_e:
            best_pair = None
            best_throughput = -float("inf")
            best_round_time = float("inf")
            for e in remaining_e:
                for c in remaining_c:
                    d = pair_cache[(e, c)]
                    if (d.throughput > best_throughput) or (
                        np.isclose(d.throughput, best_throughput) and d.round_time < best_round_time
                    ):
                        best_throughput = d.throughput
                        best_round_time = d.round_time
                        best_pair = (e, c)
            e, c = best_pair
            matching[e] = c
            remaining_e.remove(e)
            remaining_c.remove(c)

        return self._result_from_matching(K, matching, "greedy")

    def solve_hungarian_for_k(self, K: int):
        W, _, _ = self.get_weight_matrix(K)
        row_ind, col_ind = linear_sum_assignment(-W)
        matching = np.full(K, -1, dtype=int)
        matching[row_ind] = col_ind
        return self._result_from_matching(K, matching, "hungarian")

    def solve_exact_for_k(self, K: int):
        best_result = None
        best_throughput = -float("inf")
        for perm in permutations(range(K)):
            result = self._result_from_matching(K, np.array(perm, dtype=int), "exact")
            if result.total_throughput > best_throughput:
                best_throughput = result.total_throughput
                best_result = result
        return best_result

    def solve_best_over_k(self, method: str, max_clusters: Optional[int] = None, cluster_penalty: float = 0.0, exact_max_k: int = 8):
        feasible_max = min(self.N, self.M)
        Kmax = feasible_max if max_clusters is None else min(max_clusters, feasible_max)

        best_score = -float("inf")
        best_result = None
        best_k = None
        all_results = {}

        for K in range(1, Kmax + 1):
            if method == "identity":
                result = self.solve_identity_for_k(K)
            elif method == "greedy":
                result = self.solve_greedy_for_k(K)
            elif method == "hungarian":
                result = self.solve_hungarian_for_k(K)
            elif method == "exact":
                if K > exact_max_k:
                    continue
                result = self.solve_exact_for_k(K)
            else:
                raise ValueError("method must be one of: identity, greedy, hungarian, exact")

            score = result.total_throughput - cluster_penalty * K
            all_results[K] = {
                "throughput": result.total_throughput,
                "score": score,
                "system_round_time_proxy": result.system_round_time_proxy,
            }

            if score > best_score:
                best_score = score
                best_result = result
                best_k = K

        return {"best_k": best_k, "best_result": best_result, "best_score": best_score, "all_results": all_results}


def benchmark_methods(
    sampler: ScenarioSampler,
    n_scenarios: int,
    max_clusters: Optional[int] = None,
    cluster_penalty: float = 0.0,
    exact_max_k: int = 7,
    base_seed: int = 2026,
):
    rng = np.random.default_rng(base_seed)
    identities, greedies, hungarians, oracles = [], [], [], []
    identity_wins_vs_greedy = 0
    greedy_wins_vs_hungarian = 0
    hungarian_wins_vs_greedy = 0
    hungarian_wins_vs_oracle = 0
    i_gap_to_oracle, g_gap_to_oracle, h_gap_to_oracle = [], [], []
    h_gain_over_greedy, h_gain_over_identity = [], []

    for idx in range(n_scenarios):
        cfg = sampler.sample()
        solver, cfg = build_scenario_from_config(cfg, rng)

        identity = solver.solve_best_over_k("identity", max_clusters=max_clusters, cluster_penalty=cluster_penalty)["best_result"]
        greedy = solver.solve_best_over_k("greedy", max_clusters=max_clusters, cluster_penalty=cluster_penalty)["best_result"]
        hungarian = solver.solve_best_over_k("hungarian", max_clusters=max_clusters, cluster_penalty=cluster_penalty)["best_result"]

        oracle = solver.solve_best_over_k(
            "exact",
            max_clusters=min(exact_max_k, solver.N, solver.M) if max_clusters is None else min(max_clusters, exact_max_k, solver.N, solver.M),
            cluster_penalty=cluster_penalty,
            exact_max_k=exact_max_k,
        )["best_result"]

        identities.append(identity.total_throughput)
        greedies.append(greedy.total_throughput)
        hungarians.append(hungarian.total_throughput)
        oracles.append(oracle.total_throughput)

        if identity.total_throughput >= greedy.total_throughput:
            identity_wins_vs_greedy += 1
        if greedy.total_throughput >= hungarian.total_throughput:
            greedy_wins_vs_hungarian += 1
        if hungarian.total_throughput >= greedy.total_throughput:
            hungarian_wins_vs_greedy += 1
        if hungarian.total_throughput >= oracle.total_throughput:
            hungarian_wins_vs_oracle += 1

        i_gap = (oracle.total_throughput - identity.total_throughput) / max(oracle.total_throughput, 1e-9)
        g_gap = (oracle.total_throughput - greedy.total_throughput) / max(oracle.total_throughput, 1e-9)
        h_gap = (oracle.total_throughput - hungarian.total_throughput) / max(oracle.total_throughput, 1e-9)

        i_gap_to_oracle.append(i_gap)
        g_gap_to_oracle.append(g_gap)
        h_gap_to_oracle.append(h_gap)

        h_gain_over_greedy.append((hungarian.total_throughput - greedy.total_throughput) / max(greedy.total_throughput, 1e-9))
        h_gain_over_identity.append((hungarian.total_throughput - identity.total_throughput) / max(identity.total_throughput, 1e-9))

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{n_scenarios} scenarios")

    summary = {
        "avg_identity_throughput": float(np.mean(identities)),
        "avg_greedy_throughput": float(np.mean(greedies)),
        "avg_hungarian_throughput": float(np.mean(hungarians)),
        "avg_oracle_throughput": float(np.mean(oracles)),
        "identity_win_rate_vs_greedy": float(identity_wins_vs_greedy / n_scenarios),
        "greedy_win_rate_vs_hungarian": float(greedy_wins_vs_hungarian / n_scenarios),
        "hungarian_win_rate_vs_greedy": float(hungarian_wins_vs_greedy / n_scenarios),
        "hungarian_win_rate_vs_oracle": float(hungarian_wins_vs_oracle / n_scenarios),
        "avg_identity_gap_to_oracle": float(np.mean(i_gap_to_oracle)),
        "avg_greedy_gap_to_oracle": float(np.mean(g_gap_to_oracle)),
        "avg_hungarian_gap_to_oracle": float(np.mean(h_gap_to_oracle)),
        "avg_hungarian_gain_over_greedy": float(np.mean(h_gain_over_greedy)),
        "avg_hungarian_gain_over_identity": float(np.mean(h_gain_over_identity)),
    }
    return summary


def build_manual_scenario(config: ManualExperimentConfig, seed: int = 0):
    rng = np.random.default_rng(seed)

    client_blocks = []
    client_type_names = []

    for _ in range(config.num_A):
        client_blocks.append(_jitter_profile(DEVICE_A_4CORE, config.edge_time_jitter, rng))
        client_type_names.append("A")
    for _ in range(config.num_B):
        client_blocks.append(_jitter_profile(DEVICE_B_2CORE, config.edge_time_jitter, rng))
        client_type_names.append("B")
    for _ in range(config.num_C):
        client_blocks.append(_jitter_profile(DEVICE_C_1CORE, config.edge_time_jitter, rng))
        client_type_names.append("C")

    if len(client_blocks) == 0:
        raise ValueError("At least one edge device is required.")
    if config.num_cloud <= 0:
        raise ValueError("At least one cloud server is required.")

    client_layer_times = np.vstack(client_blocks)
    server_layer_times = np.vstack([
        _jitter_profile(DEVICE_CLOUD, config.cloud_time_jitter, rng)
        for _ in range(config.num_cloud)
    ])

    N = client_layer_times.shape[0]
    M = server_layer_times.shape[0]

    if config.network_rates_matrix is not None:
        network_rates = np.asarray(config.network_rates_matrix, dtype=float)
        if network_rates.shape != (N, M):
            raise ValueError(f"network_rates_matrix must have shape ({N}, {M})")
    else:
        network_rates = np.full((N, M), config.network_rate_mb_s, dtype=float)

    if config.network_overheads_matrix is not None:
        network_overheads = np.asarray(config.network_overheads_matrix, dtype=float)
        if network_overheads.shape != (N, M):
            raise ValueError(f"network_overheads_matrix must have shape ({N}, {M})")
    else:
        network_overheads = np.full((N, M), config.network_overhead_s, dtype=float)

    solver = DeterministicSimilarityAssignmentSolver(
        client_layer_times=client_layer_times,
        server_layer_times=server_layer_times,
        cut_data_sizes=CUT_DATA_SIZES_MB,
        input_data_size=config.input_data_mb,
        network_rates=network_rates,
        network_overheads=network_overheads,
    )
    solver.client_type_names = client_type_names
    return solver


def run_manual_hungarian_case(config: ManualExperimentConfig, seed: int = 0):
    solver = build_manual_scenario(config, seed=seed)

    max_clusters = config.max_clusters
    if max_clusters is None:
        max_clusters = min(solver.N, solver.M)

    identity_result = solver.solve_best_over_k(
        method="identity",
        max_clusters=max_clusters,
        cluster_penalty=config.cluster_penalty,
    )["best_result"]

    greedy_result = solver.solve_best_over_k(
        method="greedy",
        max_clusters=max_clusters,
        cluster_penalty=config.cluster_penalty,
    )["best_result"]

    hungarian_result = solver.solve_best_over_k(
        method="hungarian",
        max_clusters=max_clusters,
        cluster_penalty=config.cluster_penalty,
    )["best_result"]

    oracle_result = None
    if min(solver.N, solver.M) <= config.exact_max_k:
        oracle_result = solver.solve_best_over_k(
            method="exact",
            max_clusters=min(max_clusters, config.exact_max_k),
            cluster_penalty=config.cluster_penalty,
            exact_max_k=config.exact_max_k,
        )["best_result"]

    print_result(identity_result, solver, title="IDENTITY MATCHING RESULT")
    print_result(greedy_result, solver, title="GREEDY MATCHING RESULT")
    print_result(hungarian_result, solver, title="HUNGARIAN MATCHING RESULT")

    if oracle_result is not None:
        print_result(oracle_result, solver, title="EXACT ORACLE RESULT")

    return {
        "solver": solver,
        "identity": identity_result,
        "greedy": greedy_result,
        "hungarian": hungarian_result,
        "oracle": oracle_result,
    }


def print_result(result: OptimizationResult, solver: DeterministicSimilarityAssignmentSolver, title: str = "RESULT"):
    print("=" * 100)
    print(title)
    print(f"METHOD               : {result.method}")
    print(f"NUM CLUSTERS         : {result.num_clusters}")
    print(f"TOTAL THROUGHPUT     : {result.total_throughput:.6f}")
    print(f"SYSTEM ROUND TIME    : {result.system_round_time_proxy:.6f}")
    print("Edge labels          :", result.edge_labels.tolist())
    print("Cloud labels         :", result.cloud_labels.tolist())
    print("Matching             :", result.matching.tolist())
    print("Best cuts            :", result.best_cuts.tolist())
    print("-" * 100)

    client_types = getattr(solver, "client_type_names", ["?"] * solver.N)
    edge_groups = solver._invert_assignment(result.edge_labels, result.num_clusters)
    cloud_groups = solver._invert_assignment(result.cloud_labels, result.num_clusters)

    for e in range(result.num_clusters):
        print(f"Edge cluster {e}: clients={edge_groups[e]} types={[client_types[i] for i in edge_groups[e]]}")
    for c in range(result.num_clusters):
        print(f"Cloud cluster {c}: servers={cloud_groups[c]}")
    print("-" * 100)

    for d in result.details:
        print(f"Edge cluster {d.edge_cluster_id} <-> Cloud cluster {d.cloud_cluster_id}")
        print(f"  Clients            : {d.clients}")
        print(f"  Servers            : {d.servers}")
        print(f"  Best cut           : {d.best_cut}")
        print(f"  Producer rate      : {d.producer_rate:.6f}")
        if np.isinf(d.service_rate):
            print("  Service rate       : inf")
        else:
            print(f"  Service rate       : {d.service_rate:.6f}")
        print(f"  Throughput         : {d.throughput:.6f}")
        print(f"  Round time         : {d.round_time:.6f}")
        print("-" * 100)


#if __name__ == "__main__":
    # train_sampler, val_sampler, test_sampler = make_default_samplers()

    # print("\n=== BENCHMARK ON VALIDATION SCENARIOS ===")
    # print(benchmark_methods(
    #     sampler=val_sampler,
    #     n_scenarios=10,
    #     max_clusters=8,
    #     cluster_penalty=0.0,
    #     exact_max_k=7,
    #     base_seed=12345,
    # ))

    # print("\n=== BENCHMARK ON TEST SCENARIOS ===")
    # print(benchmark_methods(
    #     sampler=test_sampler,
    #     n_scenarios=10,
    #     max_clusters=8,
    #     cluster_penalty=0.0,
    #     exact_max_k=7,
    #     base_seed=54321,
    # ))

    # print("\n=== ONE RANDOM TEST SCENARIO ===")
    # rng = np.random.default_rng(2028)
    # cfg = test_sampler.sample()
    # solver, cfg = build_scenario_from_config(cfg, rng)

    # identity = solver.solve_best_over_k(method="identity", max_clusters=8)["best_result"]
    # greedy = solver.solve_best_over_k(method="greedy", max_clusters=8)["best_result"]
    # hungarian = solver.solve_best_over_k(method="hungarian", max_clusters=8)["best_result"]

    # print_result(identity, solver, title="IDENTITY MATCHING RESULT")
    # print_result(greedy, solver, title="GREEDY MATCHING RESULT")
    # print_result(hungarian, solver, title="HUNGARIAN MATCHING RESULT")

    ### Run manual configured scenario
    #config = ManualExperimentConfig(
    #    num_A=6,
    #    num_B=6,
    #    num_C=6,
    #    num_cloud=3,
    #    network_rate_mb_s=1000.0,
    #    network_overhead_s=0.0,
    #    edge_time_jitter=0.0,
    #    cloud_time_jitter=0.0,
    #    max_clusters=6,
    #    cluster_penalty=0.0,
    #    exact_max_k=6,
    #)

    #results = run_manual_hungarian_case(config, seed=2028)



class Server:
    def __init__(self, config):
        self.config = config
        # RabbitMQ
        self.address = config["rabbit"]["address"]
        self.username = config["rabbit"]["username"]
        self.password = config["rabbit"]["password"]
        self.virtual_host = config["rabbit"]["virtual-host"]
        
        
        self.model_name = config["server"]["model"]
        self.total_clients = config["server"]["clients"]
        self.cut_layer = config["server"]["cut-layer"]
        self.batch_size = config["server"]["batch-size"]

        global DEVICE_A_4CORE, DEVICE_B_2CORE, DEVICE_C_1CORE, DEVICE_CLOUD
        DEVICE_A_4CORE, DEVICE_B_2CORE, DEVICE_C_1CORE, DEVICE_CLOUD = \
            _select_timing_profiles(self.model_name)

        credentials = pika.PlainCredentials(self.username, self.password)
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=self.address,
                port=5672,
                virtual_host=f"{self.virtual_host}",
                credentials=credentials,
                heartbeat=0,
                blocked_connection_timeout=300
            )
        )
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue='rpc_queue')

        self.register_clients = [0 for _ in range(len(self.total_clients))]
        self.list_clients = []
        self.count_clients = 0
        self.client_assignments = {}   # {client_id: {"splits": int, "queue_name": str}}
        self.client_profile_data = {}  # {client_id_str: np.array of per-layer times}

        self.channel.basic_qos(prefetch_count=1)
        self.reply_channel = self.connection.channel()
        self.channel.basic_consume(queue='rpc_queue', on_message_callback=self.on_request)

        self.data = config["data"]
        self.compress = config["compress"]

        log_path = config["log-path"]
        self.logger = src.Log.Logger(f"{log_path}/app.log" , config["debug-mode"])
        self.logger.log_info(f"Application start. Server is waiting for {self.total_clients} clients.")
        src.Log.print_with_color(f"Application start. Server is waiting for {self.total_clients} clients.", "green")

    def _get_mode(self):
        exp = self.config.get("experiment", {})
        if exp.get("enable", True):
            return exp.get("mode", "split")
        return "split"

    def on_request(self, ch, method, _, body):
        message = pickle.loads(body)
        action = message["action"]

        if action == "REGISTER":
            client_id = message["client_id"]
            layer_id = message["layer_id"]

            if (str(client_id), layer_id) not in self.list_clients:
                self.list_clients.append((str(client_id), layer_id))

            src.Log.print_with_color(f"[<<<] Received message from client: {message}", "blue")
            if layer_id < 1 or layer_id > len(self.register_clients):
                src.Log.print_with_color(f"[!] Ignored client with unexpected layer_id={layer_id} (expected 1..{len(self.register_clients)})", "red")
                return

            layer_times = message.get("layer_times", None)
            if layer_times is not None:
                self.client_profile_data[str(client_id)] = np.array(layer_times, dtype=float)
                src.Log.print_with_color(
                    f"[Profile] Stored real profiling data from client {client_id} "
                    f"({len(layer_times)} layers, total={sum(layer_times)*1000:.1f} ms)",
                    "cyan"
                )

            self.register_clients[layer_id-1] += 1

            if self.register_clients == self.total_clients:
                src.Log.print_with_color("All clients are connected. Sending notifications.", "green")
                self.notify_clients()


        elif action == "NOTIFY":

            self.count_clients += 1

            mode = self._get_mode()

            if mode == "only_edge":

                expected_done_clients = self.total_clients[0]

            elif mode == "only_cloud":

                expected_done_clients = self.total_clients[0]

            else:

                expected_done_clients = self.total_clients[0]

            if self.count_clients == expected_done_clients:
                self.logger.log_info("Stop Inference !!!")

                self.notify_clients(start=False)

                sys.exit()
        ch.basic_ack(delivery_tag=method.delivery_tag)

    def send_to_response(self, client_id, message):
        reply_queue_name = f"reply_{client_id}"
        self.reply_channel.queue_declare(reply_queue_name, durable=False)
        src.Log.print_with_color(f"[>>>] Sent notification to client {client_id}", "red")
        self.reply_channel.basic_publish(
            exchange='',
            routing_key=reply_queue_name,
            body=message
        )

    def start(self):
        self.channel.start_consuming()

    def notify_clients(self, start=True):
        if start:
            default_splits = {
                "a": 4,
                "b": 11,
                "c": 17,
                "d": 23
            }
            if os.path.exists(f"{self.model_name}.pt"):
                src.Log.print_with_color(f"Exist {self.model_name}", "green")
            else:
                src.Log.print_with_color(f"Download {self.model_name}", "yellow")
                _ = YOLO(f"{self.model_name}.pt")


            mode = self._get_mode()

            if mode in ["only_edge", "only_cloud"]:

                splits = None

                src.Log.print_with_color(
                    f"[Benchmark] mode={mode}, skip split selection",
                    "yellow"
                )

            else:

                selected_cut = self.config["server"].get("cut-layer", "hungarian")

                if selected_cut in default_splits:
                    splits = default_splits[selected_cut]
                else:
                    splits = None

                if selected_cut == "hungarian":

                    try:
                        exp = self.config.get("experiment", {})
                        if not exp:
                            raise KeyError("Section 'experiment' missing in config.yaml — required for hungarian mode")

                        # Try to use real profiling data sent by clients at REGISTER time
                        edge_times_list = [
                            self.client_profile_data[str(cid)]
                            for cid, lid in self.list_clients
                            if lid == 1 and str(cid) in self.client_profile_data
                        ]
                        cloud_times_list = [
                            self.client_profile_data[str(cid)]
                            for cid, lid in self.list_clients
                            if lid == len(self.total_clients) and str(cid) in self.client_profile_data
                        ]
                        use_real = (
                            len(edge_times_list) == sum(1 for _, lid in self.list_clients if lid == 1) and
                            len(cloud_times_list) == sum(1 for _, lid in self.list_clients if lid == len(self.total_clients)) and
                            len(edge_times_list) > 0 and len(cloud_times_list) > 0
                        )

                        if use_real:
                            src.Log.print_with_color(
                                f"[Hungarian] Using REAL device profiles "
                                f"({len(edge_times_list)} edge, {len(cloud_times_list)} cloud)",
                                "cyan"
                            )
                            client_layer_times = np.vstack(edge_times_list)
                            server_layer_times = np.vstack(cloud_times_list)
                            N = client_layer_times.shape[0]
                            M = server_layer_times.shape[0]
                            network_rate    = float(exp.get("network_rate_mb_s", 100.0))
                            network_overhead = float(exp.get("network_overhead_s", 0.0))
                            max_clusters_cfg = exp.get("max_clusters")

                            solver = DeterministicSimilarityAssignmentSolver(
                                client_layer_times=client_layer_times,
                                server_layer_times=server_layer_times,
                                cut_data_sizes=CUT_DATA_SIZES_MB,
                                input_data_size=RAW_INPUT_MB,
                                network_rates=np.full((N, M), network_rate),
                                network_overheads=np.full((N, M), network_overhead),
                            )
                            h_out = solver.solve_best_over_k(
                                "hungarian",
                                max_clusters=max_clusters_cfg,
                            )
                            h = h_out["best_result"]
                            print_result(h, solver, title="HUNGARIAN MATCHING RESULT (real profiles)")

                        else:
                            src.Log.print_with_color(
                                "[Hungarian] Using HARDCODED device profiles (clients have no profiling cache yet)",
                                "yellow"
                            )
                            hungarian_config = ManualExperimentConfig(
                                num_A=exp["num_A"],
                                num_B=exp["num_B"],
                                num_C=exp["num_C"],
                                num_cloud=exp["num_cloud"],
                                network_rate_mb_s=exp["network_rate_mb_s"],
                                network_overhead_s=exp["network_overhead_s"],
                                max_clusters=exp["max_clusters"],
                                cluster_penalty=0.0,
                                exact_max_k=exp["exact_max_k"],
                            )
                            hungarian_results = run_manual_hungarian_case(
                                hungarian_config,
                                seed=2028
                            )
                            h = hungarian_results["hungarian"]

                        edge_labels  = h.edge_labels   # (N_edge,)
                        cloud_labels = h.cloud_labels  # (N_cloud,)
                        matching     = h.matching      # matching[k] = cloud cluster ℓ
                        best_cuts    = h.best_cuts     # best_cuts[k] = cut cho edge cluster k
                        K            = h.num_clusters

                        # inv_matching: cloud cluster ℓ → edge cluster k
                        inv_matching = {int(matching[k]): k for k in range(K)}

                        # Tách danh sách client theo layer
                        edge_ord  = [(cid, lid) for cid, lid in self.list_clients if lid == 1]
                        cloud_ord = [(cid, lid) for cid, lid in self.list_clients if lid == len(self.total_clients)]

                        # Gán per-client splits và queue_name
                        self.client_assignments = {}
                        for i, (cid, _) in enumerate(edge_ord):
                            k = int(edge_labels[i]) if i < len(edge_labels) else 0
                            self.client_assignments[cid] = {
                                "splits":     int(best_cuts[k]) + 1,
                                "queue_name": f"intermediate_queue_{k}",
                            }
                        for j, (cid, _) in enumerate(cloud_ord):
                            l = int(cloud_labels[j]) if j < len(cloud_labels) else 0
                            k = inv_matching.get(l, 0)
                            self.client_assignments[cid] = {
                                "splits":     int(best_cuts[k]) + 1,
                                "queue_name": f"intermediate_queue_{k}",
                            }

                        # splits fallback cho log
                        splits = int(best_cuts[0]) + 1 if len(best_cuts) > 0 else None

                        src.Log.print_with_color(
                            f"[Hungarian] K={K}  best_cuts={best_cuts.tolist()}  queue per cluster",
                            "green"
                        )

                    except Exception as e:
                        raise RuntimeError(f"Hungarian failed: {e}")

                elif selected_cut in default_splits:

                    src.Log.print_with_color(
                        f"[Benchmark] Fixed split {selected_cut} -> splits = {splits}",
                        "yellow"
                    )

                else:
                    raise ValueError(
                        f"Invalid cut-layer: {selected_cut}"
                    )


            file_path = f"{self.model_name}.pt"
            if os.path.exists(file_path):
                src.Log.print_with_color(f"Send model {self.model_name} to devices.", "green")
                with open(f"{self.model_name}.pt", "rb") as f:
                    file_bytes = f.read()
                    encoded = base64.b64encode(file_bytes).decode('utf-8')
            else:
                src.Log.print_with_color(f"{self.model_name} does not exist.", "yellow")
                sys.exit()

            for (client_id, layer_id) in self.list_clients:
                assignment    = self.client_assignments.get(client_id, {})
                per_splits    = assignment.get("splits",     splits)
                per_queue     = assignment.get("queue_name", "intermediate_queue")

                response = {"action": "START",
                            "message": "Server accept the connection",
                            "model": encoded,
                            "splits":     per_splits,
                            "queue_name": per_queue,
                            "batch_size": self.batch_size,
                            "num_layers": len(self.total_clients),
                            "model_name": self.model_name,
                            "data": self.data,
                            "compress": self.compress,
                            "mode": self._get_mode()}

                self.send_to_response(client_id, pickle.dumps(response))
        else:
            response = {"action": "STOP",
                        "message": "Stop inference !!!"}
            for (client_id, layer_id) in self.list_clients:
                self.send_to_response(client_id, pickle.dumps(response))
