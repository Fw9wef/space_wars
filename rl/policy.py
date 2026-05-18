"""Actor-critic with masked categorical heads."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from rl.encoding import MAX_PLANETS, NUM_SEND_MODES, OBS_DIM


def _masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.bool()
    if mask.ndim == 1:
        mask = mask.unsqueeze(0).expand_as(logits)
    out = logits.clone()
    out[~mask] = -1e9
    # If entire row masked, allow index 0 to avoid NaN
    bad = ~mask.any(dim=-1)
    if bad.any():
        out[bad, 0] = 0.0
    return out


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, hidden: int = 128) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.source_head = nn.Linear(hidden, MAX_PLANETS)
        self.target_head = nn.Linear(hidden, MAX_PLANETS)
        self.send_head = nn.Linear(hidden, NUM_SEND_MODES)
        self.value_head = nn.Linear(hidden, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(
        self,
        obs: torch.Tensor,
        source_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        src_logits = _masked_logits(self.source_head(h), source_mask)
        tgt_logits = _masked_logits(self.target_head(h), target_mask)
        send_logits = self.send_head(h)
        value = self.value_head(h).squeeze(-1)
        return src_logits, tgt_logits, send_logits, value

    def _distributions(
        self,
        obs: torch.Tensor,
        source_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> tuple[Categorical, Categorical, Categorical, torch.Tensor]:
        src_l, tgt_l, send_l, value = self.forward(obs, source_mask, target_mask)
        return Categorical(logits=src_l), Categorical(logits=tgt_l), Categorical(logits=send_l), value

    def act(
        self,
        obs_np: np.ndarray,
        source_mask: np.ndarray,
        target_mask: np.ndarray,
        *,
        deterministic: bool = False,
    ) -> tuple[tuple[int, int, int], float, float]:
        device = next(self.parameters()).device
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
        src_m = torch.as_tensor(source_mask, dtype=torch.bool, device=device).unsqueeze(0)
        tgt_m = torch.as_tensor(target_mask, dtype=torch.bool, device=device).unsqueeze(0)

        src_d, tgt_d, send_d, value = self._distributions(obs_t, src_m, tgt_m)

        if deterministic:
            a_src = int(src_d.probs.argmax(dim=-1).item())
            a_tgt = int(tgt_d.probs.argmax(dim=-1).item())
            a_send = int(send_d.probs.argmax(dim=-1).item())
        else:
            a_src = int(src_d.sample().item())
            a_tgt = int(tgt_d.sample().item())
            a_send = int(send_d.sample().item())

        logprob = (
            src_d.log_prob(torch.tensor(a_src, device=device))
            + tgt_d.log_prob(torch.tensor(a_tgt, device=device))
            + send_d.log_prob(torch.tensor(a_send, device=device))
        ).item()

        return (a_src, a_tgt, a_send), logprob, float(value.item())

    def evaluate_batch(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        source_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """actions: (B, 3) long."""
        src_d, tgt_d, send_d, value = self._distributions(obs, source_mask, target_mask)
        logprob = (
            src_d.log_prob(actions[:, 0])
            + tgt_d.log_prob(actions[:, 1])
            + send_d.log_prob(actions[:, 2])
        )
        entropy = src_d.entropy() + tgt_d.entropy() + send_d.entropy()
        return logprob, value, entropy
