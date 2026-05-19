"""Tests for mixed 2p / 4p rollout collection."""

from __future__ import annotations

import numpy as np

from rl.buffer import RolloutBuffer
from rl.graph_encoding import GraphFeatureConfig
from rl.graph_policy import GraphActorCritic
from rl.runner import RolloutCollector


def test_collect_four_player_episode():
    cfg = GraphFeatureConfig(history_steps=2, future_steps=2, include_edges=False)
    policy = GraphActorCritic(config=cfg)
    collector = RolloutCollector(
        policy,
        seed=7,
        episode_steps=30,
        four_player_fraction=1.0,
        graph_config=cfg,
    )
    buf = RolloutBuffer(16, config=cfg)
    n, ep_stats, _ = collector.collect(buf, 16, seed=7)
    assert n == 16
    assert ep_stats.num_agents == 4
    assert len(ep_stats.rewards) >= 4


def test_collect_mixed_modes():
    cfg = GraphFeatureConfig(history_steps=2, future_steps=2, include_edges=False)
    policy = GraphActorCritic(config=cfg)
    collector = RolloutCollector(
        policy,
        seed=11,
        episode_steps=20,
        four_player_fraction=0.5,
        graph_config=cfg,
    )
    buf = RolloutBuffer(32, config=cfg)
    n, _, _ = collector.collect(buf, 32, seed=11)
    assert n == 32
