"""Episode-scoped edge travel cache and fleet arrival projections."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from orbit_dynamics import (
    CENTER,
    ROTATION_RADIUS_LIMIT,
    intercept_angle_for_target,
    rollout_first_event,
)

from rl.graph_constants import MAX_BASE_PLANETS, MAX_NODES, SHIP_BUCKETS
from rl.encoding import obs_get

Configuration = dict[str, float] | Any
PlanetRow = list


@dataclass
class EdgeBucketResult:
    reachable: bool = False
    travel_ticks: int = 0
    angle: float = 0.0
    event_code: int = 3

    def as_features(self, *, tick_norm: float) -> np.ndarray:
        """reachable, ticks_norm, sin(angle), cos(angle) per bucket slice (4 floats)."""
        t = float(self.travel_ticks) * tick_norm if self.reachable else 0.0
        if self.reachable:
            return np.array(
                [1.0, t, math.sin(self.angle), math.cos(self.angle)],
                dtype=np.float32,
            )
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)


@dataclass
class FleetProjection:
    fleet_id: int
    owner: int
    ships: int
    target_slot: int
    eta_ticks: int
    spawn_step: int

    def remaining_eta(self, current_step: int) -> int:
        elapsed = max(0, current_step - self.spawn_step)
        return max(0, self.eta_ticks - elapsed)


class EdgeTravelCache:
    """Lazy cache: (tick, source_slot, target_slot, ship_bucket) -> EdgeBucketResult."""

    def __init__(
        self,
        *,
        episode_steps: int = 500,
        max_ticks: int = 400,
        coarse: int = 8,
    ) -> None:
        self.episode_steps = episode_steps
        self.max_ticks = max_ticks
        self.coarse = coarse
        self._cache: dict[tuple[int, int, int, int], EdgeBucketResult] = {}
        self._slot_mapping: Any = None
        self._initial_planets: list[PlanetRow] = []
        self._angular_velocity: float = 0.0
        self._configuration: Configuration = {}
        self._comet_ids: set[int] = set()

    def reset(self, obs: Any, *, episode_steps: int | None = None) -> Any:
        from rl.graph_encoding import build_slot_mapping

        if episode_steps is not None:
            self.episode_steps = episode_steps
        self._cache.clear()
        self._slot_mapping = build_slot_mapping(obs)
        self._initial_planets = list(obs_get(obs, "initial_planets", []) or [])
        self._angular_velocity = float(obs_get(obs, "angular_velocity", 0.0))
        self._configuration = obs_get(obs, "configuration") or {}
        cids = obs_get(obs, "comet_planet_ids", []) or []
        self._comet_ids = {int(x) for x in cids}
        return self._slot_mapping

    def invalidate_comet_edges(self) -> None:
        """Drop cached entries that may depend on comet positions."""
        keys = [k for k in self._cache if self._uses_comet_slot(k[1]) or self._uses_comet_slot(k[2])]
        for k in keys:
            del self._cache[k]

    def _uses_comet_slot(self, slot: int) -> bool:
        return slot >= MAX_BASE_PLANETS

    def _planet_row_at_tick(
        self,
        pid: int,
        tick: int,
        planets_rows: list[PlanetRow],
        comets: list[dict[str, Any]],
        comet_planet_ids: list[int],
    ) -> PlanetRow | None:
        by_id = {int(p[0]): list(p) for p in planets_rows}
        if pid in by_id:
            row = by_id[pid]
        else:
            init = {int(p[0]): p for p in self._initial_planets}
            if pid not in init:
                return None
            row = list(init[pid])

        if pid in self._comet_ids:
            return row

        dx = float(row[2]) - CENTER
        dy = float(row[3]) - CENTER
        r = math.hypot(dx, dy)
        if r + float(row[4]) >= ROTATION_RADIUS_LIMIT:
            return row

        init = {int(p[0]): p for p in self._initial_planets}
        ip = init.get(pid)
        if ip is None:
            return row
        dx0 = float(ip[2]) - CENTER
        dy0 = float(ip[3]) - CENTER
        r0 = math.hypot(dx0, dy0)
        ang = math.atan2(dy0, dx0) + self._angular_velocity * float(tick)
        row[2] = CENTER + r0 * math.cos(ang)
        row[3] = CENTER + r0 * math.sin(ang)
        return row

    def _query_bucket(
        self,
        tick: int,
        source_slot: int,
        target_slot: int,
        ship_bucket: int,
        obs: Any,
    ) -> EdgeBucketResult:
        key = (tick, source_slot, target_slot, ship_bucket)
        if key in self._cache:
            return self._cache[key]

        assert self._slot_mapping is not None
        sm = self._slot_mapping
        src_pid = sm.slot_to_planet_id(source_slot)
        tgt_pid = sm.slot_to_planet_id(target_slot)
        result = EdgeBucketResult()
        if src_pid < 0 or tgt_pid < 0 or source_slot == target_slot:
            self._cache[key] = result
            return result

        planets_rows = list(obs_get(obs, "planets", []) or [])
        comets = list(obs_get(obs, "comets") or [])
        cids = list(obs_get(obs, "comet_planet_ids") or [])

        src_row = self._planet_row_at_tick(src_pid, tick, planets_rows, comets, cids)
        tgt_row = self._planet_row_at_tick(tgt_pid, tick, planets_rows, comets, cids)
        if src_row is None or tgt_row is None:
            self._cache[key] = result
            return result

        ships = int(ship_bucket)
        angle, feasible = intercept_angle_for_target(
            src_row,
            tgt_row,
            ships,
            planets_rows,
            self._initial_planets,
            self._angular_velocity,
            tick,
            comets,
            cids,
            self._configuration,
            coarse=self.coarse,
            max_ticks=self.max_ticks,
        )
        if angle is None:
            angle = math.atan2(
                float(tgt_row[3]) - float(src_row[3]),
                float(tgt_row[2]) - float(src_row[2]),
            )

        code, _extra, ticks = rollout_first_event(
            src_row,
            float(angle),
            ships,
            copy.deepcopy(planets_rows),
            copy.deepcopy(self._initial_planets),
            self._angular_velocity,
            tick,
            copy.deepcopy(comets),
            cids,
            tgt_pid,
            self._configuration,
            max_ticks=self.max_ticks,
        )
        result.event_code = code
        result.angle = float(angle)
        result.reachable = code == 0 and feasible
        result.travel_ticks = int(ticks) if result.reachable else 0
        self._cache[key] = result
        return result

    def edge_features_for_pair(
        self,
        tick: int,
        source_slot: int,
        target_slot: int,
        obs: Any,
    ) -> np.ndarray:
        """All ship buckets: shape (NUM_SHIP_BUCKETS * 4,)."""
        tick_norm = 1.0 / max(1, self.episode_steps)
        parts = []
        for bucket in SHIP_BUCKETS:
            r = self._query_bucket(tick, source_slot, target_slot, bucket, obs)
            parts.append(r.as_features(tick_norm=tick_norm))
        return np.concatenate(parts).astype(np.float32)

    def fill_edge_tensor(self, tick: int, obs: Any, edge_tensor: np.ndarray) -> None:
        """Write edge features into edge_tensor[MAX_NODES, MAX_NODES, EDGE_FEAT_DIM]."""
        assert self._slot_mapping is not None
        for s in range(MAX_NODES):
            for t in range(MAX_NODES):
                if s == t:
                    continue
                edge_tensor[s, t] = self.edge_features_for_pair(tick, s, t, obs)


class FleetProjectionCache:
    """Estimate fleet destination and ETA once per fleet_id."""

    def __init__(self, *, max_ticks: int = 400) -> None:
        self.max_ticks = max_ticks
        self._projections: dict[int, FleetProjection] = {}
        self._slot_mapping: Any = None
        self._configuration: Configuration = {}
        self._initial_planets: list[PlanetRow] = []
        self._angular_velocity: float = 0.0

    def reset(self, slot_mapping: Any, obs: Any) -> None:
        self._projections.clear()
        self._slot_mapping = slot_mapping
        self._configuration = obs_get(obs, "configuration") or {}
        self._initial_planets = list(obs_get(obs, "initial_planets", []) or [])
        self._angular_velocity = float(obs_get(obs, "angular_velocity", 0.0))

    def update(self, obs: Any, slot_mapping: Any) -> None:
        self._slot_mapping = slot_mapping
        step = int(obs_get(obs, "step", 0))
        planets_rows = list(obs_get(obs, "planets", []) or [])
        comets = list(obs_get(obs, "comets") or [])
        cids = list(obs_get(obs, "comet_planet_ids") or [])
        fleets = list(obs_get(obs, "fleets", []) or [])
        active_ids = {int(f[0]) for f in fleets}

        stale = [fid for fid in self._projections if fid not in active_ids]
        for fid in stale:
            del self._projections[fid]

        by_id = {int(p[0]): p for p in planets_rows}
        dummy_from = [0, -1, 0.0, 0.0, 1.0, 1, 1]

        for fleet in fleets:
            fid = int(fleet[0])
            if fid in self._projections:
                continue
            owner = int(fleet[1])
            fx, fy = float(fleet[2]), float(fleet[3])
            angle = float(fleet[4])
            ships = int(fleet[6])
            spawn_step = step

            best_slot = -1
            best_ticks = self.max_ticks

            for slot in range(MAX_NODES):
                pid = slot_mapping.slot_to_planet_id(slot)
                if pid < 0:
                    continue
                code, detail, ticks = rollout_first_event(
                    dummy_from,
                    angle,
                    ships,
                    copy.deepcopy(planets_rows),
                    copy.deepcopy(self._initial_planets),
                    self._angular_velocity,
                    step,
                    copy.deepcopy(comets),
                    cids,
                    pid,
                    self._configuration,
                    max_ticks=self.max_ticks,
                    fleet_xy=(fx, fy),
                )
                if code in (0, 2) and ticks < best_ticks:
                    hit_pid = int(detail) if detail is not None else pid
                    if code == 0:
                        hit_pid = pid
                    hit_slot = slot_mapping.slot_for_planet(hit_pid)
                    if hit_slot >= 0:
                        best_ticks = ticks
                        best_slot = hit_slot

            if best_slot < 0:
                continue

            self._projections[fid] = FleetProjection(
                fleet_id=fid,
                owner=owner,
                ships=ships,
                target_slot=best_slot,
                eta_ticks=best_ticks,
                spawn_step=spawn_step,
            )

    def incoming_by_slot(
        self,
        current_step: int,
        player: int,
        horizon: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (friendly, enemy) arrays shape (horizon, MAX_NODES) ship counts
        indexed by arrival offset 0..horizon-1 from current step.
        """
        friendly = np.zeros((horizon, MAX_NODES), dtype=np.float32)
        enemy = np.zeros((horizon, MAX_NODES), dtype=np.float32)
        for proj in self._projections.values():
            if proj.target_slot < 0:
                continue
            remaining = proj.remaining_eta(current_step)
            if remaining >= horizon:
                continue
            ships = float(proj.ships) / 100.0
            if proj.owner == player:
                friendly[remaining, proj.target_slot] += ships
            else:
                enemy[remaining, proj.target_slot] += ships
        return friendly, enemy
