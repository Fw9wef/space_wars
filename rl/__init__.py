"""PPO self-play baseline for Orbit Wars."""

from rl.encoding import MAX_PLANETS, NUM_SEND_MODES, OBS_DIM, encode_observation
from rl.action import decode_action, noop_action
from rl.rewards import RewardConfig, shaped_step_reward, terminal_reward
from rl.policy import ActorCritic
from rl.runner import RolloutCollector

__all__ = [
    "MAX_PLANETS",
    "NUM_SEND_MODES",
    "OBS_DIM",
    "encode_observation",
    "decode_action",
    "noop_action",
    "RewardConfig",
    "shaped_step_reward",
    "terminal_reward",
    "ActorCritic",
    "RolloutCollector",
]
