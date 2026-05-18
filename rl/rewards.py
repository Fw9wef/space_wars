"""Reward shaping and terminal rewards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rl.encoding import obs_get


@dataclass
class RewardConfig:
    alpha: float = 1.0
    beta: float = 5.0
    alpha_final: float = 0.0
    beta_final: float = 0.0
    decay_fraction: float = 0.5

    def coeffs(self, progress: float) -> tuple[float, float]:
        """Linear decay of shaping coeffs; progress in [0, 1]."""
        if self.decay_fraction <= 0:
            return self.alpha, self.beta
        t = min(1.0, progress / self.decay_fraction)
        alpha = self.alpha + t * (self.alpha_final - self.alpha)
        beta = self.beta + t * (self.beta_final - self.beta)
        return alpha, beta


@dataclass
class PlayerRewardState:
    ship_total: int = 0
    owned_planets: int = 0


def _ship_total(obs: Any, player: int) -> int:
    total = 0
    for p in obs_get(obs, "planets", []) or []:
        if int(p[1]) == player:
            total += int(p[5])
    for f in obs_get(obs, "fleets", []) or []:
        if int(f[1]) == player:
            total += int(f[6])
    return total


def _owned_planets(obs: Any, player: int) -> int:
    return sum(1 for p in obs_get(obs, "planets", []) or [] if int(p[1]) == player)


def init_reward_state(obs: Any, player: int) -> PlayerRewardState:
    return PlayerRewardState(
        ship_total=_ship_total(obs, player),
        owned_planets=_owned_planets(obs, player),
    )


def shaped_step_reward(
    obs_after: Any,
    state: PlayerRewardState,
    player: int,
    *,
    alpha: float,
    beta: float,
) -> tuple[float, PlayerRewardState]:
    ships = _ship_total(obs_after, player)
    owned = _owned_planets(obs_after, player)
    d_ships = ships - state.ship_total
    d_owned = owned - state.owned_planets
    reward = alpha * (d_ships / 100.0) + beta * float(d_owned)
    new_state = PlayerRewardState(ship_total=ships, owned_planets=owned)
    return reward, new_state


def terminal_reward(env_final_state: list, player: int) -> float:
    return float(env_final_state[player].reward)
