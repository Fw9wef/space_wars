"""
Continuous-time fleet intercept: min arrival time with disk contact and path constraints.

Not engine-faithful (no discrete ticks, no swept_pair_hit, no planet list order).
Use for fast heading estimates; verify in-game if exact rollout behavior matters.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from kaggle_environments.envs.orbit_wars.orbit_wars import (
    CENTER,
    SUN_RADIUS,
    point_to_segment_distance,
)

from orbit_dynamics import BOARD_SIZE, ROTATION_RADIUS_LIMIT, fleet_speed

PlanetRow = list
Configuration = dict[str, float] | Any

__all__ = ["intercept_angle_for_target"]

_INF = float("inf")
_TAU_EPS = 1e-6


def _get_cfg(configuration: Configuration, key: str, default: float) -> float:
    if isinstance(configuration, dict):
        return float(configuration.get(key, default))
    return float(getattr(configuration, key, default))


def _normalize_angle(a: float) -> float:
    while a < 0.0:
        a += 2 * math.pi
    while a >= 2 * math.pi:
        a -= 2 * math.pi
    return a


def _aim_naive(from_planet: PlanetRow, target: PlanetRow) -> float:
    return math.atan2(
        float(target[3]) - float(from_planet[3]),
        float(target[2]) - float(from_planet[2]),
    )


class _InterceptModel:
    """Planet motion + ray fleet geometry for one launch scenario."""

    def __init__(
        self,
        from_planet: PlanetRow,
        target: PlanetRow,
        fleet_ships: int,
        planets: Sequence[PlanetRow],
        initial_planets: Sequence[PlanetRow],
        angular_velocity: float,
        start_step: int,
        comets: list[dict[str, Any]],
        comet_planet_ids: list[int],
        configuration: Configuration,
        max_ticks: int,
        path_sample_dt: float,
    ) -> None:
        self.from_planet = from_planet
        self.target = target
        self.target_id = int(target[0])
        self.fleet_ships = fleet_ships
        self.omega = float(angular_velocity)
        self.start_step = int(start_step)
        self.max_ticks = float(max_ticks)
        self.path_sample_dt = max(path_sample_dt, 1e-3)
        self.sun_radius = _get_cfg(configuration, "sunRadius", SUN_RADIUS)
        self.v = fleet_speed(fleet_ships, configuration)

        self.cx = float(from_planet[2])
        self.cy = float(from_planet[3])
        self.s0 = float(from_planet[4]) + 0.1
        self.r_target = float(target[4])

        self.planets_by_id = {int(p[0]): p for p in planets}
        self.initial_by_id = {int(p[0]): p for p in initial_planets}
        self.comet_pid_set = set(int(x) for x in comet_planet_ids)
        self._orbital_ids: set[int] = set()
        for p in planets:
            pid = int(p[0])
            if pid in self.comet_pid_set:
                continue
            initial_p = self.initial_by_id.get(pid)
            if initial_p is None:
                continue
            dx = float(initial_p[2]) - CENTER
            dy = float(initial_p[3]) - CENTER
            r = math.hypot(dx, dy)
            if r + float(p[4]) < ROTATION_RADIUS_LIMIT:
                self._orbital_ids.add(pid)

        self._comet_keyframes: dict[int, list[tuple[float, float]]] = {}
        self._comet_lookup: dict[int, tuple[dict[str, Any], int]] = {}
        for group in comets:
            for i, pid in enumerate(group["planet_ids"]):
                pid = int(pid)
                self._comet_lookup[pid] = (group, i)
                self._comet_keyframes[pid] = self._build_comet_keyframes(group, i, pid)

        self._blocker_ids = [
            int(p[0]) for p in planets if int(p[0]) not in (int(from_planet[0]), self.target_id)
        ]

    def _build_comet_keyframes(
        self, group: dict[str, Any], path_i: int, pid: int
    ) -> list[tuple[float, float]]:
        planet = self.planets_by_id[pid]
        path = group["paths"][path_i]
        base_idx = int(group["path_index"])
        keyframes: list[tuple[float, float]] = [(float(planet[2]), float(planet[3]))]
        pos = keyframes[0]
        for k in range(int(self.max_ticks) + 1):
            idx = base_idx + 1 + k
            if idx >= len(path):
                keyframes.append(pos)
            else:
                pos = (float(path[idx][0]), float(path[idx][1]))
                keyframes.append(pos)
        return keyframes

    def fleet_xy(self, theta: float, tau: float) -> tuple[float, float]:
        s = self.s0 + self.v * tau
        c, sn = math.cos(theta), math.sin(theta)
        return self.cx + s * c, self.cy + s * sn

    def _orbital_xy(self, pid: int, tau: float) -> tuple[float, float]:
        initial_p = self.initial_by_id[pid]
        p = self.planets_by_id[pid]
        dx = float(initial_p[2]) - CENTER
        dy = float(initial_p[3]) - CENTER
        r = math.hypot(dx, dy)
        alpha = math.atan2(dy, dx)
        ang = alpha + self.omega * (self.start_step + tau)
        return CENTER + r * math.cos(ang), CENTER + r * math.sin(ang)

    def _comet_xy(self, pid: int, tau: float) -> tuple[float, float]:
        kf = self._comet_keyframes[pid]
        if tau <= 0.0:
            return kf[0]
        if tau >= len(kf) - 1:
            return kf[-1]
        k = int(tau)
        frac = tau - k
        x0, y0 = kf[k]
        x1, y1 = kf[k + 1]
        return x0 + (x1 - x0) * frac, y0 + (y1 - y0) * frac

    def planet_xy(self, pid: int, tau: float) -> tuple[float, float]:
        if pid in self.comet_pid_set:
            return self._comet_xy(pid, tau)
        if pid in self._orbital_ids:
            return self._orbital_xy(pid, tau)
        p = self.planets_by_id[pid]
        return float(p[2]), float(p[3])

    def _dist_to_target(self, theta: float, tau: float) -> float:
        fx, fy = self.fleet_xy(theta, tau)
        px, py = self.planet_xy(self.target_id, tau)
        return math.hypot(fx - px, fy - py)

    def _inside_target(self, theta: float, tau: float) -> bool:
        return self._dist_to_target(theta, tau) <= self.r_target

    def time_to_target_contact(self, theta: float) -> float:
        """First tau >= 0 with ||F(tau)-P_target(tau)|| <= R_target, or inf."""
        if self._inside_target(theta, 0.0):
            return 0.0

        hi = self.path_sample_dt
        while hi <= self.max_ticks and not self._inside_target(theta, hi):
            hi += self.path_sample_dt
            if hi > self.max_ticks + self.path_sample_dt:
                return _INF

        if hi > self.max_ticks and not self._inside_target(theta, min(hi, self.max_ticks)):
            return _INF

        lo = max(0.0, hi - self.path_sample_dt)
        for _ in range(40):
            if hi - lo < _TAU_EPS:
                break
            mid = 0.5 * (lo + hi)
            if self._inside_target(theta, mid):
                hi = mid
            else:
                lo = mid
        return hi

    def _oob_at(self, theta: float, tau: float) -> bool:
        x, y = self.fleet_xy(theta, tau)
        return not (0.0 <= x <= BOARD_SIZE and 0.0 <= y <= BOARD_SIZE)

    def _sun_hit_on_segment(self, theta: float, tau: float) -> bool:
        if tau <= 0.0:
            return False
        p0 = self.fleet_xy(theta, 0.0)
        p1 = self.fleet_xy(theta, tau)
        return point_to_segment_distance((CENTER, CENTER), p0, p1) < self.sun_radius

    def _first_blocker_tau(self, theta: float, t_hit: float) -> float:
        """Earliest tau in [0, t_hit] where a blocker triggers, or inf."""
        if t_hit <= 0.0:
            if self._oob_at(theta, 0.0):
                return 0.0
            return _INF

        first = _INF
        tau = 0.0
        while tau <= t_hit + _TAU_EPS:
            if self._oob_at(theta, tau):
                return min(first, tau)
            if self._sun_hit_on_segment(theta, tau):
                first = min(first, tau)
            fx, fy = self.fleet_xy(theta, tau)
            for pid in self._blocker_ids:
                p = self.planets_by_id[pid]
                rn = float(p[4])
                px, py = self.planet_xy(pid, tau)
                if math.hypot(fx - px, fy - py) <= rn:
                    first = min(first, tau)
            tau += self.path_sample_dt
        return first

    def is_feasible(self, theta: float, t_hit: float) -> bool:
        if not math.isfinite(t_hit) or t_hit == _INF:
            return False
        t_block = self._first_blocker_tau(theta, t_hit)
        return t_hit < t_block - _TAU_EPS

    def lead_theta(self) -> float:
        """Initial heading from continuous lead pursuit along the launch ray."""
        px0, py0 = self.planet_xy(self.target_id, 0.0)

        def reach_gap(tau: float) -> float:
            px, py = self.planet_xy(self.target_id, tau)
            dist = math.hypot(px - self.cx, py - self.cy)
            s_need = self.s0 + self.v * tau
            return dist - s_need - self.r_target

        if reach_gap(0.0) <= 0.0:
            return math.atan2(py0 - self.cy, px0 - self.cx)

        hi = self.path_sample_dt
        while hi <= self.max_ticks and reach_gap(hi) > 0.0:
            hi += self.path_sample_dt
        if hi > self.max_ticks and reach_gap(self.max_ticks) > 0.0:
            return math.atan2(py0 - self.cy, px0 - self.cx)

        lo = max(0.0, hi - self.path_sample_dt)
        for _ in range(40):
            if hi - lo < _TAU_EPS:
                break
            mid = 0.5 * (lo + hi)
            if reach_gap(mid) <= 0.0:
                hi = mid
            else:
                lo = mid
        tau_lead = hi
        px, py = self.planet_xy(self.target_id, tau_lead)
        return math.atan2(py - self.cy, px - self.cx)

    def evaluate(self, theta: float) -> tuple[float, bool]:
        """Return (t_hit, feasible) for this heading."""
        t_hit = self.time_to_target_contact(theta)
        return t_hit, self.is_feasible(theta, t_hit)


def _golden_section_min(
    lo: float, hi: float, f, iters: int = 12
) -> float:
    for _ in range(iters):
        m1 = lo + (hi - lo) * 0.382
        m2 = lo + (hi - lo) * 0.618
        if f(m1) < f(m2):
            hi = m2
        else:
            lo = m1
    return 0.5 * (lo + hi)


def intercept_angle_for_target(
    from_planet: PlanetRow,
    target: PlanetRow,
    fleet_ships: int,
    planets: Sequence[PlanetRow],
    initial_planets: Sequence[PlanetRow],
    angular_velocity: float,
    start_step: int,
    comets: list[dict[str, Any]],
    comet_planet_ids: list[int],
    configuration: Configuration,
    *,
    coarse: int = 8,
    max_ticks: int = 400,
    path_sample_dt: float = 0.25,
) -> tuple[float | None, bool]:
    """
    Minimize continuous arrival time t_hit(theta) with ||F(t)-P_target(t)|| <= R_target
    and path clearance from sun, OOB, and other planets.

    Returns (angle, feasible). If no feasible heading exists, returns a fallback angle
    (best approximate or naive atan2) with feasible=False.
    """
    model = _InterceptModel(
        from_planet,
        target,
        fleet_ships,
        planets,
        initial_planets,
        angular_velocity,
        start_step,
        comets,
        comet_planet_ids,
        configuration,
        max_ticks,
        path_sample_dt,
    )

    def cost(theta: float) -> float:
        t_hit, ok = model.evaluate(theta)
        return t_hit if ok else _INF

    theta0 = model.lead_theta()
    naive = _aim_naive(from_planet, target)
    span = 2 * math.pi / max(coarse, 1)
    candidates = [
        theta0,
        theta0 + span,
        theta0 - span,
        naive,
    ]

    best_theta: float | None = None
    best_t = _INF
    best_feasible_theta: float | None = None
    best_feasible_t = _INF

    for a in candidates:
        a = _normalize_angle(a)
        t_hit, ok = model.evaluate(a)
        if ok and t_hit < best_feasible_t:
            best_feasible_t, best_feasible_theta = t_hit, a
        if t_hit < best_t:
            best_t, best_theta = t_hit, a

    seed = best_feasible_theta if best_feasible_theta is not None else best_theta
    if seed is None:
        return None, False

    lo = seed - span
    hi = seed + span

    refined = _golden_section_min(lo, hi, cost)
    refined = _normalize_angle(refined)
    t_ref, ok_ref = model.evaluate(refined)
    if ok_ref and t_ref < best_feasible_t:
        best_feasible_t, best_feasible_theta = t_ref, refined

    if best_feasible_theta is not None:
        return best_feasible_theta, True

    if best_theta is not None and math.isfinite(best_t):
        return best_theta, False

    return naive, False
