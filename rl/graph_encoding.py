"""Graph observation encoding for Orbit Wars RL."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from kaggle_environments.envs.orbit_wars.orbit_wars import CENTER, ROTATION_RADIUS_LIMIT

from rl.graph_cache import EdgeTravelCache, FleetProjectionCache
from rl.graph_constants import (
    BASE_NODE_FEAT_DIM,
    DEFAULT_FUTURE_STEPS,
    DEFAULT_HISTORY_STEPS,
    EDGE_FEAT_DIM,
    GLOBAL_FEAT_DIM,
    MAX_BASE_PLANETS,
    MAX_COMETS,
    MAX_NODES,
    NUM_SEND_MODES,
)
from rl.encoding import obs_get

# re-export for callers
__all__ = [
    "MAX_NODES",
    "NUM_SEND_MODES",
    "encode_graph_observation",
    "GraphObs",
    "GraphFeatureState",
    "GraphFeatureConfig",
]


@dataclass(frozen=True)
class GraphFeatureConfig:
    history_steps: int = DEFAULT_HISTORY_STEPS
    future_steps: int = DEFAULT_FUTURE_STEPS
    # Full N×N edge tensor requires many intercept rollouts per step; disable for speed.
    include_edges: bool = False

    @property
    def node_feat_dim(self) -> int:
        return BASE_NODE_FEAT_DIM + 2 * self.history_steps + 2 * self.future_steps


@dataclass
class SlotMapping:
    slot_planet_ids: np.ndarray
    planet_to_slot: dict[int, int]
    node_valid: np.ndarray
    comet_slot_for_pid: dict[int, int]

    def slot_to_planet_id(self, slot: int) -> int:
        if slot < 0 or slot >= MAX_NODES:
            return -1
        return int(self.slot_planet_ids[slot])

    def slot_for_planet(self, pid: int) -> int:
        return self.planet_to_slot.get(int(pid), -1)


def build_slot_mapping(obs: Any) -> SlotMapping:
    """Assign base planets by sorted id; reserve last 4 slots for comets."""
    planets_rows = list(obs_get(obs, "planets", []) or [])
    comet_ids = sorted(int(x) for x in (obs_get(obs, "comet_planet_ids", []) or []))
    comet_id_set = set(comet_ids)
    base_ids = sorted(int(p[0]) for p in planets_rows if int(p[0]) not in comet_id_set)

    slot_ids = np.full(MAX_NODES, -1, dtype=np.int64)
    valid = np.zeros(MAX_NODES, dtype=bool)
    pid_to_slot: dict[int, int] = {}
    comet_slot_map: dict[int, int] = {}

    for i, pid in enumerate(base_ids[:MAX_BASE_PLANETS]):
        slot_ids[i] = pid
        valid[i] = True
        pid_to_slot[pid] = i

    for j, pid in enumerate(comet_ids[:MAX_COMETS]):
        slot = MAX_BASE_PLANETS + j
        slot_ids[slot] = pid
        valid[slot] = True
        pid_to_slot[pid] = slot
        comet_slot_map[pid] = slot

    return SlotMapping(
        slot_planet_ids=slot_ids,
        planet_to_slot=pid_to_slot,
        node_valid=valid,
        comet_slot_for_pid=comet_slot_map,
    )


@dataclass
class GraphObs:
    nodes: np.ndarray
    edges: np.ndarray
    global_features: np.ndarray
    source_mask: np.ndarray
    target_mask: np.ndarray
    node_valid: np.ndarray
    edge_valid: np.ndarray
    slot_planet_ids: np.ndarray

    @property
    def info(self) -> dict[str, np.ndarray]:
        return {
            "source_mask": self.source_mask,
            "target_mask": self.target_mask,
            "planet_valid": self.node_valid,
            "slot_planet_ids": self.slot_planet_ids,
            "edge_valid": self.edge_valid,
        }


@dataclass
class GraphFeatureState:
    """Per-player episode state: history, caches, last comet signature."""

    config: GraphFeatureConfig = field(default_factory=GraphFeatureConfig)
    player: int = 0
    edge_cache: EdgeTravelCache = field(default_factory=EdgeTravelCache)
    fleet_cache: FleetProjectionCache = field(default_factory=FleetProjectionCache)
    slot_mapping: SlotMapping | None = None
    _history: dict[int, deque[tuple[float, float]]] = field(default_factory=dict)
    _last_comet_sig: tuple[Any, ...] = ()

    def reset(self, obs: Any, *, episode_steps: int = 500) -> None:
        self.player = int(obs_get(obs, "player", 0))
        self._history.clear()
        self.slot_mapping = self.edge_cache.reset(obs, episode_steps=episode_steps)
        self.fleet_cache.reset(self.slot_mapping, obs)
        self._last_comet_sig = _comet_signature(obs)

    def _ensure_history_slot(self, slot: int) -> deque[tuple[float, float]]:
        if slot not in self._history:
            self._history[slot] = deque(maxlen=self.config.history_steps)
        return self._history[slot]

    def update_caches(self, obs: Any) -> None:
        assert self.slot_mapping is not None
        sig = _comet_signature(obs)
        if sig != self._last_comet_sig:
            self.edge_cache.invalidate_comet_edges()
            self._last_comet_sig = sig
        self.slot_mapping = build_slot_mapping(obs)
        self.fleet_cache.update(obs, self.slot_mapping)

    def push_history(self, obs: Any) -> None:
        assert self.slot_mapping is not None
        player = int(obs_get(obs, "player", self.player))
        planets_rows = list(obs_get(obs, "planets", []) or [])
        by_id = {int(p[0]): p for p in planets_rows}

        for slot in range(MAX_NODES):
            pid = self.slot_mapping.slot_to_planet_id(slot)
            if pid < 0:
                continue
            p = by_id.get(pid)
            if p is None:
                continue
            owner = int(p[1])
            ships = float(p[5]) / 100.0
            if owner == player:
                rel = 1.0
            elif owner < 0:
                rel = 0.0
            else:
                rel = -1.0
            self._ensure_history_slot(slot).append((ships, rel))

    def encode(self, obs: Any) -> GraphObs:
        if self.slot_mapping is None:
            self.reset(obs)
        self.update_caches(obs)

        cfg = self.config
        player = int(obs_get(obs, "player", self.player))
        step = int(obs_get(obs, "step", 0))
        omega = float(obs_get(obs, "angular_velocity", 0.0))
        cfg_obj = obs_get(obs, "configuration", None)
        if isinstance(cfg_obj, dict):
            episode_steps = int(cfg_obj.get("episodeSteps", 500))
        elif cfg_obj is not None and hasattr(cfg_obj, "episodeSteps"):
            episode_steps = int(getattr(cfg_obj, "episodeSteps"))
        else:
            episode_steps = 500
        tick_norm = 1.0 / max(1, episode_steps)

        sm = self.slot_mapping
        assert sm is not None

        nodes = np.zeros((MAX_NODES, cfg.node_feat_dim), dtype=np.float32)
        edges = np.zeros((MAX_NODES, MAX_NODES, EDGE_FEAT_DIM), dtype=np.float32)
        edge_valid = np.zeros((MAX_NODES, MAX_NODES), dtype=bool)
        source_mask = np.zeros(MAX_NODES, dtype=bool)
        target_mask = np.zeros(MAX_NODES, dtype=bool)

        planets_rows = list(obs_get(obs, "planets", []) or [])
        by_id = {int(p[0]): p for p in planets_rows}
        initial_by_id = {int(p[0]): p for p in obs_get(obs, "initial_planets", []) or []}

        n_mine = 0
        n_enemy = 0

        for slot in range(MAX_NODES):
            pid = sm.slot_to_planet_id(slot)
            if pid < 0 or not sm.node_valid[slot]:
                continue
            p = by_id.get(pid)
            if p is None:
                continue

            owner = int(p[1])
            x, y = float(p[2]) / 100.0, float(p[3]) / 100.0
            ships = float(p[5]) / 100.0
            radius = float(p[4]) / 10.0
            production = float(p[6]) / 5.0

            is_mine = owner == player
            is_enemy = owner != player and owner >= 0
            is_neutral = owner < 0

            if is_mine:
                n_mine += 1
            elif is_enemy or is_neutral:
                n_enemy += 1

            ip = initial_by_id.get(pid, p)
            dx = float(ip[2]) - CENTER
            dy = float(ip[3]) - CENTER
            orbit_r = math.hypot(dx, dy) / 50.0
            is_rotating = 1.0 if orbit_r * 50.0 + float(p[4]) < ROTATION_RADIUS_LIMIT else 0.0

            base = [
                x,
                y,
                1.0 if is_mine else 0.0,
                1.0 if is_enemy else 0.0,
                1.0 if is_neutral else 0.0,
                ships,
                radius,
                production,
                orbit_r,
                is_rotating,
            ]

            hist_feats: list[float] = []
            hist = self._history.get(slot, deque())
            hist_list = list(hist)
            n_hist = len(hist_list)
            for k in range(cfg.history_steps):
                idx = n_hist - cfg.history_steps + k
                if 0 <= idx < n_hist:
                    s, rel = hist_list[idx]
                    hist_feats.extend([s, rel])
                else:
                    hist_feats.extend([0.0, 0.0])

            friendly, enemy = self.fleet_cache.incoming_by_slot(
                step, player, cfg.future_steps
            )
            fut_feats: list[float] = []
            for k in range(cfg.future_steps):
                fut_feats.append(float(friendly[k, slot]))
                fut_feats.append(float(enemy[k, slot]))

            # passive production projection (owned planets only)
            if is_mine:
                prod = float(p[6])
                for k in range(cfg.future_steps):
                    fut_feats[2 * k] += prod * (k + 1) / 100.0

            nodes[slot] = np.array(base + hist_feats + fut_feats, dtype=np.float32)

            if is_mine and int(p[5]) > 0:
                source_mask[slot] = True
            if owner != player:
                target_mask[slot] = True

        for s in range(MAX_NODES):
            for t in range(MAX_NODES):
                if s == t or not sm.node_valid[s] or not sm.node_valid[t]:
                    continue
                if cfg.include_edges:
                    edges[s, t] = self.edge_cache.edge_features_for_pair(step, s, t, obs)
                    edge_valid[s, t] = True

        global_features = np.array(
            [
                step * tick_norm,
                omega,
                n_mine / MAX_NODES,
                n_enemy / MAX_NODES,
                float(player) / 3.0,
                float(sm.node_valid.sum()) / MAX_NODES,
            ],
            dtype=np.float32,
        )

        graph = GraphObs(
            nodes=nodes,
            edges=edges,
            global_features=global_features,
            source_mask=source_mask,
            target_mask=target_mask,
            node_valid=sm.node_valid.copy(),
            edge_valid=edge_valid,
            slot_planet_ids=sm.slot_planet_ids.copy(),
        )
        self.push_history(obs)
        return graph


def _comet_signature(obs: Any) -> tuple[Any, ...]:
    comets = obs_get(obs, "comets") or []
    cids = tuple(sorted(int(x) for x in (obs_get(obs, "comet_planet_ids", []) or [])))
    parts = []
    for g in comets:
        parts.append((int(g.get("path_index", 0)), tuple(int(x) for x in g.get("planet_ids", []))))
    return (cids, tuple(parts))


def encode_graph_observation(
    obs: Any,
    state: GraphFeatureState | None = None,
    *,
    config: GraphFeatureConfig | None = None,
    episode_steps: int = 500,
) -> tuple[GraphObs, GraphFeatureState]:
    if state is None:
        state = GraphFeatureState(config=config or GraphFeatureConfig())
        state.reset(obs, episode_steps=episode_steps)
    return state.encode(obs), state
