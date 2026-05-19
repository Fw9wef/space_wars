"""
Orbit Wars RL agent (Kaggle entry point).

Place checkpoint.pt next to this file or set ORBIT_RL_CHECKPOINT env var.
"""

from __future__ import annotations

import os
from pathlib import Path

from rl.runner import RLAgent

_ROOT = Path(__file__).resolve().parent
_DEFAULT_CKPT = _ROOT / "runs" / "ppo_graph_v0" / "checkpoint.pt"
_agent: RLAgent | None = None


def _checkpoint_path() -> Path:
    env_path = os.environ.get("ORBIT_RL_CHECKPOINT")
    if env_path:
        return Path(env_path)
    return _DEFAULT_CKPT


def agent(obs):
    global _agent
    if _agent is None:
        ckpt = _checkpoint_path()
        if not ckpt.is_file():
            raise FileNotFoundError(
                f"RL checkpoint not found: {ckpt}. Train with train_selfplay.py or set ORBIT_RL_CHECKPOINT."
            )
        _agent = RLAgent.from_checkpoint(str(ckpt), device="cpu", deterministic=True)
    return _agent(obs)
