"""Environment stepping and rollout collection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch
from kaggle_environments import make

from rl.action import decode_action
from rl.buffer import RolloutBuffer
from rl.graph_encoding import GraphFeatureConfig, GraphFeatureState, encode_graph_observation
from rl.graph_policy import GraphActorCritic
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
    num_agents: int = 2
    rewards: list[float] = field(default_factory=list)
    terminals: list[float] = field(default_factory=list)

    @property
    def reward_p0(self) -> float:
        return self.rewards[0] if self.rewards else 0.0

    @property
    def reward_p1(self) -> float:
        return self.rewards[1] if len(self.rewards) > 1 else 0.0

    @property
    def terminal_p0(self) -> float:
        return self.terminals[0] if self.terminals else 0.0

    @property
    def terminal_p1(self) -> float:
        return self.terminals[1] if len(self.terminals) > 1 else 0.0


def format_episode_stats(ep_stats: EpisodeStats) -> str:
    n = ep_stats.num_agents
    parts = [f"ep_r{i}={ep_stats.rewards[i]:.2f}" for i in range(n)]
    suffix = " 4p" if n == 4 else ""
    return " ".join(parts) + suffix


class RolloutCollector:
    """Collect PPO transitions from self-play (shared policy), 2p or 4p FFA."""

    def __init__(
        self,
        policy: GraphActorCritic,
        *,
        seed: int = 0,
        episode_steps: int = 500,
        reward_config: RewardConfig | None = None,
        training_progress: float = 0.0,
        four_player_fraction: float = 0.5,
        graph_config: GraphFeatureConfig | None = None,
        debug: bool = False,
    ) -> None:
        self.policy = policy
        self.seed = seed
        self.episode_steps = episode_steps
        self.reward_config = reward_config or RewardConfig()
        self.training_progress = training_progress
        self.four_player_fraction = four_player_fraction
        self.graph_config = graph_config or GraphFeatureConfig()
        self._rng = np.random.default_rng(seed)
        self.debug = debug
        self._env = None
        self._num_agents = 2
        self._graph_states: list[GraphFeatureState] = []

    def _make_env(self) -> None:
        self._env = make(
            "orbit_wars",
            configuration={"seed": self.seed, "episodeSteps": self.episode_steps},
            debug=self.debug,
        )

    def _sample_num_agents(self) -> int:
        if self.four_player_fraction <= 0.0:
            return 2
        if self.four_player_fraction >= 1.0:
            return 4
        return 4 if self._rng.random() < self.four_player_fraction else 2

    def _init_graph_states(self, obs_list: list[Any]) -> None:
        self._graph_states = []
        for obs in obs_list:
            st = GraphFeatureState(config=self.graph_config)
            st.reset(obs, episode_steps=self.episode_steps)
            self._graph_states.append(st)

    def _reset(self, seed: int | None = None, num_agents: int | None = None) -> list[Any]:
        if self._env is None:
            self._make_env()
        if seed is not None:
            self.seed = seed
            self._env = None
            self._make_env()
        if num_agents is None:
            num_agents = self._sample_num_agents()
        self._num_agents = num_agents
        self._env.reset(num_agents=num_agents)
        obs_list = [observation_for_player(self._env.state, i) for i in range(num_agents)]
        self._init_graph_states(obs_list)
        return obs_list

    def _policy_moves(
        self,
        raw_obs: Any,
        graph_state: GraphFeatureState,
        *,
        deterministic: bool,
    ) -> tuple:
        graph_obs, _ = encode_graph_observation(
            raw_obs, graph_state, episode_steps=self.episode_steps
        )
        action, logprob, value = self.policy.act(graph_obs, deterministic=deterministic)
        moves = decode_action(
            raw_obs,
            action,
            graph_obs.slot_planet_ids,
        )
        return moves, graph_obs, action, logprob, value

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
        obs_list = self._reset(seed)
        num_agents = self._num_agents
        rstates = [init_reward_state(obs, i) for i, obs in enumerate(obs_list)]
        ep_stats = EpisodeStats(
            num_agents=num_agents,
            rewards=[0.0] * 4,
            terminals=[0.0] * 4,
        )
        collected = 0
        last_value = 0.0

        while collected < n_steps and not self._env.done:
            step_out = [
                self._policy_moves(
                    obs, self._graph_states[i], deterministic=deterministic
                )
                for i, obs in enumerate(obs_list)
            ]
            moves = [out[0] for out in step_out]
            values = [out[4] for out in step_out]
            last_value = float(sum(values)) / len(values)

            self._env.step(moves)

            obs_list_next = [
                observation_for_player(self._env.state, i) for i in range(num_agents)
            ]

            rewards = []
            for i in range(num_agents):
                r_i, rstates[i] = shaped_step_reward(
                    obs_list_next[i], rstates[i], i, alpha=alpha, beta=beta
                )
                rewards.append(r_i)

            done = self._env.done
            if done:
                final = self._env.state
                for i in range(num_agents):
                    t_i = terminal_reward(final, i)
                    rewards[i] += t_i
                    ep_stats.terminals[i] = t_i

            for i in range(num_agents):
                moves_i, graph_obs_i, act_i, lp_i, val_i = step_out[i]
                buffer.add(graph_obs_i, act_i, lp_i, val_i, rewards[i], done)
                ep_stats.rewards[i] += rewards[i]
                collected += 1

            ep_stats.steps += 1
            obs_list = obs_list_next

            if done:
                num_agents = self._sample_num_agents()
                ep_stats.num_agents = num_agents
                obs_list = self._reset(num_agents=num_agents)
                rstates = [init_reward_state(obs, i) for i, obs in enumerate(obs_list)]

            if progress_callback and log_interval > 0 and ep_stats.steps % log_interval == 0:
                progress_callback(ep_stats.steps, collected)

        if not self._env.done and self._graph_states:
            _, _, _, _, last_value = self._policy_moves(
                obs_list[0], self._graph_states[0], deterministic=True
            )

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
    """Callable Kaggle agent backed by GraphActorCritic checkpoint."""

    def __init__(
        self,
        policy: GraphActorCritic,
        *,
        deterministic: bool = True,
        graph_config: GraphFeatureConfig | None = None,
        episode_steps: int = 500,
    ) -> None:
        self.policy = policy
        self.deterministic = deterministic
        self.graph_config = graph_config or GraphFeatureConfig()
        self.episode_steps = episode_steps
        self._graph_state: GraphFeatureState | None = None
        self.policy.eval()

    def __call__(self, obs: Any, configuration: Any = None) -> list:
        if self._graph_state is None:
            self._graph_state = GraphFeatureState(config=self.graph_config)
            self._graph_state.reset(obs, episode_steps=self.episode_steps)
        graph_obs, _ = encode_graph_observation(
            obs, self._graph_state, episode_steps=self.episode_steps
        )
        action, _, _ = self.policy.act(graph_obs, deterministic=self.deterministic)
        return decode_action(obs, action, graph_obs.slot_planet_ids)

    def reset_episode(self) -> None:
        self._graph_state = None

    @classmethod
    def from_checkpoint(
        cls, path: str, device: str = "cpu", deterministic: bool = True
    ) -> RLAgent:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        graph_config = GraphFeatureConfig()
        if "graph_config" in ckpt:
            gc = ckpt["graph_config"]
            graph_config = GraphFeatureConfig(
                history_steps=int(gc.get("history_steps", 5)),
                future_steps=int(gc.get("future_steps", 5)),
                include_edges=bool(gc.get("include_edges", False)),
            )
        policy = GraphActorCritic(config=graph_config)
        policy.load_state_dict(ckpt["policy"])
        policy.to(device)
        return cls(policy, deterministic=deterministic, graph_config=graph_config)
