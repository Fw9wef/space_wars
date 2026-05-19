"""PPO update for graph policy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from rl.buffer import RolloutBatch
from rl.graph_policy import GraphActorCritic


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr: float = 3e-4
    update_epochs: int = 4
    minibatch_size: int = 256


class PPOTrainer:
    def __init__(self, policy: GraphActorCritic, config: PPOConfig | None = None) -> None:
        self.policy = policy
        self.config = config or PPOConfig()
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=self.config.lr, eps=1e-5)

    def update(self, batch: RolloutBatch) -> dict[str, float]:
        cfg = self.config
        device = next(self.policy.parameters()).device

        nodes = torch.as_tensor(batch.nodes, dtype=torch.float32, device=device)
        edges = torch.as_tensor(batch.edges, dtype=torch.float32, device=device)
        global_f = torch.as_tensor(batch.global_features, dtype=torch.float32, device=device)
        actions = torch.as_tensor(batch.actions, dtype=torch.long, device=device)
        old_logprob = torch.as_tensor(batch.logprobs, dtype=torch.float32, device=device)
        advantages = torch.as_tensor(batch.advantages, dtype=torch.float32, device=device)
        returns = torch.as_tensor(batch.returns, dtype=torch.float32, device=device)
        node_valid = torch.as_tensor(batch.node_valid, dtype=torch.bool, device=device)
        source_mask = torch.as_tensor(batch.source_mask, dtype=torch.bool, device=device)
        target_mask = torch.as_tensor(batch.target_mask, dtype=torch.bool, device=device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = batch.size
        indices = np.arange(n)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}
        n_updates = 0

        for _ in range(cfg.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, n, cfg.minibatch_size):
                mb = indices[start : start + cfg.minibatch_size]
                logprob, value, entropy = self.policy.evaluate_batch(
                    nodes[mb],
                    edges[mb],
                    global_f[mb],
                    node_valid[mb],
                    actions[mb],
                    source_mask[mb],
                    target_mask[mb],
                )
                ratio = torch.exp(logprob - old_logprob[mb])
                mb_adv = advantages[mb]
                mb_ret = returns[mb]
                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                policy_loss = torch.max(pg1, pg2).mean()

                value_loss = 0.5 * ((value - mb_ret) ** 2).mean()
                ent_loss = entropy.mean()

                loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (old_logprob[mb] - logprob).mean().item()
                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += ent_loss.item()
                stats["approx_kl"] += approx_kl
                n_updates += 1

        for k in stats:
            stats[k] /= max(1, n_updates)
        return stats
