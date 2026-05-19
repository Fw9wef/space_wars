"""Smoke test: graph rollout + one PPO update."""

from __future__ import annotations

import torch

from rl.buffer import RolloutBuffer
from rl.graph_encoding import GraphFeatureConfig
from rl.graph_policy import GraphActorCritic
from rl.ppo import PPOConfig, PPOTrainer
from rl.runner import RolloutCollector


def test_graph_ppo_smoke():
    graph_config = GraphFeatureConfig(history_steps=2, future_steps=2, include_edges=False)
    policy = GraphActorCritic(config=graph_config)
    collector = RolloutCollector(
        policy,
        seed=0,
        episode_steps=25,
        four_player_fraction=0.0,
        graph_config=graph_config,
    )
    buf = RolloutBuffer(32, config=graph_config)
    n, _, last_v = collector.collect(buf, 32, seed=0)
    assert n == 32
    batch = buf.compute_gae(last_v, 0.99, 0.95)
    trainer = PPOTrainer(policy, PPOConfig(minibatch_size=16, update_epochs=1))
    stats = trainer.update(batch)
    assert "policy_loss" in stats
    assert torch.isfinite(torch.tensor(stats["policy_loss"]))
