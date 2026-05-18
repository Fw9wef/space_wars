"""Environment stepping and rollout collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from kaggle_environments import make

from rl.action import decode_action
from rl.buffer import RolloutBuffer
from rl.encoding import encode_observation
from rl.policy import ActorCritic
from rl.rewards import (
    PlayerRewardState,
    RewardConfig,
    init_reward_state,
    shaped_step_reward,
    terminal_reward,
)


def observation_for_player(env_state: list, player: int) -> Any:
    return env_state[player].observation


@dataclass
class EpisodeStats:
    steps: int = 0
    reward_p0: float = 0.0
    reward_p1: float = 0.0
    terminal_p0: float = 0.0
    terminal_p1: float = 0.0


class RolloutCollector:
    """Collect PPO transitions from 2-player self-play (shared policy)."""

    def __init__(
        self,
        policy: ActorCritic,
        *,
        seed: int = 0,
        episode_steps: int = 500,
        reward_config: RewardConfig | None = None,
        training_progress: float = 0.0,
        debug: bool = False,
    ) -> None:
        self.policy = policy
        self.seed = seed
        self.episode_steps = episode_steps
        self.reward_config = reward_config or RewardConfig()
        self.training_progress = training_progress
        self.debug = debug
        self._env = None

    def _make_env(self) -> None:
        self._env = make(
            "orbit_wars",
            configuration={"seed": self.seed, "episodeSteps": self.episode_steps},
            debug=self.debug,
        )

    def _reset(self, seed: int | None = None) -> tuple[Any, Any]:
        if self._env is None:
            self._make_env()
        if seed is not None:
            self.seed = seed
            self._env = None
            self._make_env()
        self._env.reset(num_agents=2)
        obs0 = observation_for_player(self._env.state, 0)
        obs1 = observation_for_player(self._env.state, 1)
        return obs0, obs1

    def _policy_moves(self, raw_obs: Any, *, deterministic: bool) -> tuple:
        obs_vec, enc_info = encode_observation(raw_obs)
        action, logprob, value = self.policy.act(
            obs_vec,
            enc_info["source_mask"],
            enc_info["target_mask"],
            deterministic=deterministic,
        )
        moves = decode_action(
            raw_obs,
            action,
            enc_info["slot_planet_ids"],
        )
        return moves, obs_vec, enc_info, action, logprob, value

    def collect(
        self,
        buffer: RolloutBuffer,
        n_steps: int,
        *,
        seed: int | None = None,
        deterministic: bool = False,
        log_interval: int = 0,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> tuple[int, EpisodeStats, float]:
        """
        Fill buffer up to n_steps transitions (both players count separately).

        Returns (steps_collected, last_episode_stats, bootstrap_value).
        """
        alpha, beta = self.reward_config.coeffs(self.training_progress)
        obs0, obs1 = self._reset(seed)
        rstate0 = init_reward_state(obs0, 0)
        rstate1 = init_reward_state(obs1, 1)
        ep_stats = EpisodeStats()
        collected = 0
        last_value = 0.0

        while collected < n_steps and not self._env.done:
            moves0, vec0, info0, act0, lp0, val0 = self._policy_moves(obs0, deterministic=deterministic)
            moves1, vec1, info1, act1, lp1, val1 = self._policy_moves(obs1, deterministic=deterministic)
            last_value = (val0 + val1) / 2.0

            self._env.step([moves0, moves1])

            obs0_next = observation_for_player(self._env.state, 0)
            obs1_next = observation_for_player(self._env.state, 1)

            r0, rstate0 = shaped_step_reward(obs0_next, rstate0, 0, alpha=alpha, beta=beta)
            r1, rstate1 = shaped_step_reward(obs1_next, rstate1, 1, alpha=alpha, beta=beta)

            done = self._env.done
            if done:
                final = self._env.state
                t0 = terminal_reward(final, 0)
                t1 = terminal_reward(final, 1)
                r0 += t0
                r1 += t1
                ep_stats.terminal_p0 = t0
                ep_stats.terminal_p1 = t1

            buffer.add(vec0, act0, lp0, val0, r0, done, info0["source_mask"], info0["target_mask"])
            buffer.add(vec1, act1, lp1, val1, r1, done, info1["source_mask"], info1["target_mask"])
            collected += 2
            ep_stats.reward_p0 += r0
            ep_stats.reward_p1 += r1
            ep_stats.steps += 1

            obs0, obs1 = obs0_next, obs1_next

            if done:
                obs0, obs1 = self._reset()
                rstate0 = init_reward_state(obs0, 0)
                rstate1 = init_reward_state(obs1, 1)

            if progress_callback and log_interval > 0 and ep_stats.steps % log_interval == 0:
                progress_callback(ep_stats.steps, collected)

        if not self._env.done:
            _, _, _, _, _, last_value = self._policy_moves(obs0, deterministic=True)

        return collected, ep_stats, last_value


def run_match(
    agent0: Callable,
    agent1: Callable,
    *,
    seed: int = 42,
    episode_steps: int = 500,
    debug: bool = False,
) -> tuple[list, int]:
    """Run one episode; agents are callables agent(obs) -> moves."""
    env = make(
        "orbit_wars",
        configuration={"seed": seed, "episodeSteps": episode_steps},
        debug=debug,
    )
    env.reset(num_agents=2)
    while not env.done:
        actions = [agent0(env.state[0].observation), agent1(env.state[1].observation)]
        env.step(actions)
    winner = 0 if env.state[0].reward > env.state[1].reward else 1
    if env.state[0].reward == env.state[1].reward:
        winner = -1
    return env.state, winner


class RLAgent:
    """Callable Kaggle agent backed by ActorCritic checkpoint."""

    def __init__(self, policy: ActorCritic, *, deterministic: bool = True) -> None:
        self.policy = policy
        self.deterministic = deterministic
        self.policy.eval()

    def __call__(self, obs: Any, configuration: Any = None) -> list:
        obs_vec, enc_info = encode_observation(obs)
        action, _, _ = self.policy.act(
            obs_vec,
            enc_info["source_mask"],
            enc_info["target_mask"],
            deterministic=self.deterministic,
        )
        return decode_action(obs, action, enc_info["slot_planet_ids"])

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu", deterministic: bool = True) -> RLAgent:
        ckpt = torch.load(path, map_location=device)
        policy = ActorCritic()
        policy.load_state_dict(ckpt["policy"])
        policy.to(device)
        return cls(policy, deterministic=deterministic)
