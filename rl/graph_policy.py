"""Graph actor-critic with dense message passing."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from rl.graph_constants import (
    EDGE_FEAT_DIM,
    GLOBAL_FEAT_DIM,
    NUM_SEND_MODES,
)
from rl.graph_encoding import GraphFeatureConfig
from rl.policy import _masked_logits


class GraphActorCritic(nn.Module):
    def __init__(
        self,
        *,
        config: GraphFeatureConfig | None = None,
        hidden: int = 128,
        edge_hidden: int = 32,
    ) -> None:
        super().__init__()
        self.config = config or GraphFeatureConfig()
        node_dim = self.config.node_feat_dim
        self.node_in = nn.Linear(node_dim, hidden)
        self.edge_in = nn.Linear(EDGE_FEAT_DIM, edge_hidden)
        self.msg = nn.Linear(hidden + edge_hidden, hidden)
        self.node_out = nn.Linear(hidden * 2, hidden)
        self.global_in = nn.Linear(GLOBAL_FEAT_DIM, hidden)
        self.source_head = nn.Linear(hidden, 1)
        self.target_head = nn.Linear(hidden, 1)
        self.send_head = nn.Linear(hidden, NUM_SEND_MODES)
        self.value_head = nn.Linear(hidden, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def _encode_graph(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        global_features: torch.Tensor,
        node_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """nodes (B,N,F), edges (B,N,N,E), global (B,G) -> node_h (B,N,H), graph_h (B,H)."""
        h = torch.relu(self.node_in(nodes))
        e = torch.relu(self.edge_in(edges))

        valid_f = node_valid.float().unsqueeze(-1)
        denom = valid_f.sum(dim=1, keepdim=True).clamp(min=1.0)

        agg = torch.zeros_like(h)
        b, n, _ = h.shape
        for i in range(n):
            hi = h[:, i : i + 1, :].expand(-1, n, -1)
            msg_in = torch.cat([hi, e[:, i, :, :]], dim=-1)
            m = torch.relu(self.msg(msg_in))
            m = m * valid_f
            agg[:, i, :] = m.sum(dim=1) / denom.squeeze(-1)

        h2 = torch.relu(self.node_out(torch.cat([h, agg], dim=-1)))
        h2 = h2 * valid_f

        g = torch.relu(self.global_in(global_features))
        graph_h = (h2 * valid_f).sum(dim=1) / denom.squeeze(-1) + g
        return h2, graph_h

    def forward(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        global_features: torch.Tensor,
        node_valid: torch.Tensor,
        source_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        node_h, graph_h = self._encode_graph(nodes, edges, global_features, node_valid)
        src_logits = _masked_logits(self.source_head(node_h).squeeze(-1), source_mask)
        tgt_logits = _masked_logits(self.target_head(node_h).squeeze(-1), target_mask)
        send_logits = self.send_head(graph_h)
        value = self.value_head(graph_h).squeeze(-1)
        return src_logits, tgt_logits, send_logits, value

    def _distributions(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        global_features: torch.Tensor,
        node_valid: torch.Tensor,
        source_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> tuple[Categorical, Categorical, Categorical, torch.Tensor]:
        src_l, tgt_l, send_l, value = self.forward(
            nodes, edges, global_features, node_valid, source_mask, target_mask
        )
        return Categorical(logits=src_l), Categorical(logits=tgt_l), Categorical(logits=send_l), value

    def act(
        self,
        graph_obs: Any,
        *,
        deterministic: bool = False,
    ) -> tuple[tuple[int, int, int], float, float]:
        device = next(self.parameters()).device
        nodes = torch.as_tensor(graph_obs.nodes, dtype=torch.float32, device=device).unsqueeze(0)
        edges = torch.as_tensor(graph_obs.edges, dtype=torch.float32, device=device).unsqueeze(0)
        global_f = torch.as_tensor(
            graph_obs.global_features, dtype=torch.float32, device=device
        ).unsqueeze(0)
        node_valid = torch.as_tensor(graph_obs.node_valid, dtype=torch.bool, device=device).unsqueeze(
            0
        )
        src_m = torch.as_tensor(graph_obs.source_mask, dtype=torch.bool, device=device).unsqueeze(0)
        tgt_m = torch.as_tensor(graph_obs.target_mask, dtype=torch.bool, device=device).unsqueeze(0)

        src_d, tgt_d, send_d, value = self._distributions(
            nodes, edges, global_f, node_valid, src_m, tgt_m
        )

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
        nodes: torch.Tensor,
        edges: torch.Tensor,
        global_features: torch.Tensor,
        node_valid: torch.Tensor,
        actions: torch.Tensor,
        source_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        src_d, tgt_d, send_d, value = self._distributions(
            nodes, edges, global_features, node_valid, source_mask, target_mask
        )
        logprob = (
            src_d.log_prob(actions[:, 0])
            + tgt_d.log_prob(actions[:, 1])
            + send_d.log_prob(actions[:, 2])
        )
        entropy = src_d.entropy() + tgt_d.entropy() + send_d.entropy()
        return logprob, value, entropy
