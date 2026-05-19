"""Rollout buffer with GAE for graph observations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from rl.graph_constants import EDGE_FEAT_DIM, GLOBAL_FEAT_DIM, MAX_NODES
from rl.graph_encoding import GraphFeatureConfig, GraphObs


@dataclass
class RolloutBatch:
    nodes: np.ndarray
    edges: np.ndarray
    global_features: np.ndarray
    actions: np.ndarray
    logprobs: np.ndarray
    values: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    node_valid: np.ndarray
    source_mask: np.ndarray
    target_mask: np.ndarray
    advantages: np.ndarray | None = None
    returns: np.ndarray | None = None

    @property
    def size(self) -> int:
        return int(self.nodes.shape[0])


def _node_feat_dim(config: GraphFeatureConfig) -> int:
    return config.node_feat_dim


class RolloutBuffer:
    def __init__(self, capacity: int, *, config: GraphFeatureConfig | None = None) -> None:
        self.capacity = capacity
        self.config = config or GraphFeatureConfig()
        nd = _node_feat_dim(self.config)
        self.ptr = 0
        self.nodes = np.zeros((capacity, MAX_NODES, nd), dtype=np.float32)
        self.edges = np.zeros((capacity, MAX_NODES, MAX_NODES, EDGE_FEAT_DIM), dtype=np.float32)
        self.global_features = np.zeros((capacity, GLOBAL_FEAT_DIM), dtype=np.float32)
        self.actions = np.zeros((capacity, 3), dtype=np.int64)
        self.logprobs = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.node_valid = np.zeros((capacity, MAX_NODES), dtype=bool)
        self.source_mask = np.zeros((capacity, MAX_NODES), dtype=bool)
        self.target_mask = np.zeros((capacity, MAX_NODES), dtype=bool)

    def add(
        self,
        graph_obs: GraphObs,
        action: tuple[int, int, int],
        logprob: float,
        value: float,
        reward: float,
        done: bool,
    ) -> None:
        i = self.ptr
        self.nodes[i] = graph_obs.nodes
        self.edges[i] = graph_obs.edges
        self.global_features[i] = graph_obs.global_features
        self.actions[i] = action
        self.logprobs[i] = logprob
        self.values[i] = value
        self.rewards[i] = reward
        self.dones[i] = float(done)
        self.node_valid[i] = graph_obs.node_valid
        self.source_mask[i] = graph_obs.source_mask
        self.target_mask[i] = graph_obs.target_mask
        self.ptr += 1

    def ready(self) -> bool:
        return self.ptr >= self.capacity

    def compute_gae(self, last_value: float, gamma: float, gae_lambda: float) -> RolloutBatch:
        n = self.ptr
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        next_value = last_value
        for t in reversed(range(n)):
            mask = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * mask - self.values[t]
            last_gae = delta + gamma * gae_lambda * mask * last_gae
            advantages[t] = last_gae
            next_value = self.values[t]
        returns = advantages + self.values[:n]
        return RolloutBatch(
            nodes=self.nodes[:n].copy(),
            edges=self.edges[:n].copy(),
            global_features=self.global_features[:n].copy(),
            actions=self.actions[:n].copy(),
            logprobs=self.logprobs[:n].copy(),
            values=self.values[:n].copy(),
            rewards=self.rewards[:n].copy(),
            dones=self.dones[:n].copy(),
            node_valid=self.node_valid[:n].copy(),
            source_mask=self.source_mask[:n].copy(),
            target_mask=self.target_mask[:n].copy(),
            advantages=advantages,
            returns=returns,
        )

    def reset(self) -> None:
        self.ptr = 0
