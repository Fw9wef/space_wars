"""Rollout buffer with GAE."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from rl.encoding import MAX_PLANETS, OBS_DIM


@dataclass
class RolloutBatch:
    obs: np.ndarray
    actions: np.ndarray
    logprobs: np.ndarray
    values: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    source_mask: np.ndarray
    target_mask: np.ndarray
    advantages: np.ndarray | None = None
    returns: np.ndarray | None = None

    @property
    def size(self) -> int:
        return int(self.obs.shape[0])


class RolloutBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.ptr = 0
        self.obs = np.zeros((capacity, OBS_DIM), dtype=np.float32)
        self.actions = np.zeros((capacity, 3), dtype=np.int64)
        self.logprobs = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.source_mask = np.zeros((capacity, MAX_PLANETS), dtype=bool)
        self.target_mask = np.zeros((capacity, MAX_PLANETS), dtype=bool)

    def add(
        self,
        obs: np.ndarray,
        action: tuple[int, int, int],
        logprob: float,
        value: float,
        reward: float,
        done: bool,
        source_mask: np.ndarray,
        target_mask: np.ndarray,
    ) -> None:
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.logprobs[i] = logprob
        self.values[i] = value
        self.rewards[i] = reward
        self.dones[i] = float(done)
        self.source_mask[i] = source_mask
        self.target_mask[i] = target_mask
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
            obs=self.obs[:n].copy(),
            actions=self.actions[:n].copy(),
            logprobs=self.logprobs[:n].copy(),
            values=self.values[:n].copy(),
            rewards=self.rewards[:n].copy(),
            dones=self.dones[:n].copy(),
            source_mask=self.source_mask[:n].copy(),
            target_mask=self.target_mask[:n].copy(),
            advantages=advantages,
            returns=returns,
        )

    def reset(self) -> None:
        self.ptr = 0
