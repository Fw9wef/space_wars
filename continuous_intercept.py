"""
Continuous-time fleet intercept: min arrival time with disk contact and path constraints.

Not engine-faithful (no discrete ticks, no swept_pair_hit, no planet list order).
Use for fast heading estimates; verify in-game if exact rollout behavior matters.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from kaggle_environments.envs.orbit_wars.orbit_wars import CENTER, SUN_RADIUS

from orbit_dynamics import BOARD_SIZE, ROTATION_RADIUS_LIMIT, fleet_speed

PlanetRow = list
Configuration = dict[str, float] | Any

__all__ = ["intercept_angle_for_target"]

_INF = float("inf")
_TAU_EPS = 1e-6
_BISECT_ITERS = 25


def _get_cfg(configuration: Configuration, key: str, default: float) -> float:
    if isinstance(configuration, dict):
        return float(configuration.get(key, default))
    return float(getattr(configuration, key, default))


def _normalize_angle(a: float) -> float:
    return float(a % (2 * math.pi))


def _aim_naive(from_planet: PlanetRow, target: PlanetRow) -> float:
    return math.atan2(
        float(target[3]) - float(from_planet[3]),
        float(target[2]) - float(from_planet[2]),
    )


def _bisect_first_nonpositive(lo: float, hi: float, f) -> float:
    if f(lo) <= 0.0:
        return lo
    if f(hi) > 0.0:
        return _INF
    for _ in range(_BISECT_ITERS):
        if hi - lo < _TAU_EPS:
            break
        mid = 0.5 * (lo + hi)
        if f(mid) <= 0.0:
            hi = mid
        else:
            lo = mid
    return hi


def _point_segment_distance(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> float:
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    denom = abx * abx + aby * aby
    if denom < 1e-18:
        return math.hypot(apx, apy)
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / denom))
    return math.hypot(px - (ax + t * abx), py - (ay + t * aby))


def _sample_comet_on_tau(keyframes: np.ndarray, tau: np.ndarray) -> np.ndarray:
    tau = np.asarray(tau, dtype=np.float64)
    k = np.floor(tau).astype(np.int64)
    k = np.clip(k, 0, len(keyframes) - 2)
    frac = (tau - k)[:, np.newaxis]
    return (1.0 - frac) * keyframes[k] + frac * keyframes[k + 1]


def _build_comet_keyframes(
    group: dict[str, Any],
    path_i: int,
    pid: int,
    planets_by_id: dict[int, PlanetRow],
    max_ticks: int,
) -> np.ndarray:
    planet = planets_by_id[pid]
    path = group["paths"][path_i]
    base_idx = int(group["path_index"])
    n = max_ticks + 1
    kf = np.empty((n + 1, 2), dtype=np.float64)
    kf[0] = (float(planet[2]), float(planet[3]))
    pos = kf[0].copy()
    for k in range(1, n + 1):
        idx = base_idx + 1 + (k - 1)
        if idx >= len(path):
            kf[k] = pos
        else:
            pos = np.array((float(path[idx][0]), float(path[idx][1])), dtype=np.float64)
            kf[k] = pos
    return kf


def _trajectory_overlaps_board(xy: np.ndarray, radius: float) -> bool:
    xmin = float(xy[:, 0].min()) - radius
    xmax = float(xy[:, 0].max()) + radius
    ymin = float(xy[:, 1].min()) - radius
    ymax = float(xy[:, 1].max()) + radius
    return not (xmax < 0.0 or xmin > BOARD_SIZE or ymax < 0.0 or ymin > BOARD_SIZE)


class _InterceptModel:
    """Planet motion + ray fleet geometry for one launch scenario (NumPy tables)."""

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
        dt: float,
    ) -> None:
        self.target_id = int(target[0])
        self.omega = float(angular_velocity)
        self.start_step = int(start_step)
        self.max_ticks = float(max_ticks)
        self.dt = max(float(dt), 1e-3)
        self.sun_radius = _get_cfg(configuration, "sunRadius", SUN_RADIUS)
        self.v = fleet_speed(fleet_ships, configuration)

        self.cx = float(from_planet[2])
        self.cy = float(from_planet[3])
        self.s0 = float(from_planet[4]) + 0.1
        self.r_target = float(target[4])

        self.tau = np.arange(0.0, self.max_ticks + self.dt, self.dt, dtype=np.float64)

        planets_by_id = {int(p[0]): p for p in planets}
        initial_by_id = {int(p[0]): p for p in initial_planets}
        comet_pid_set = {int(x) for x in comet_planet_ids}
        from_id = int(from_planet[0])

        orbital_ids: set[int] = set()
        for p in planets:
            pid = int(p[0])
            if pid in comet_pid_set:
                continue
            initial_p = initial_by_id.get(pid)
            if initial_p is None:
                continue
            dx = float(initial_p[2]) - CENTER
            dy = float(initial_p[3]) - CENTER
            r = math.hypot(dx, dy)
            if r + float(p[4]) < ROTATION_RADIUS_LIMIT:
                orbital_ids.add(pid)

        blocker_ids = [
            int(p[0]) for p in planets if int(p[0]) not in (from_id, self.target_id)
        ]
        needed_pids = {self.target_id, *blocker_ids}

        comet_keyframes: dict[int, np.ndarray] = {}
        for group in comets:
            for i, pid in enumerate(group["planet_ids"]):
                pid = int(pid)
                if pid not in needed_pids or pid not in comet_pid_set:
                    continue
                comet_keyframes[pid] = _build_comet_keyframes(
                    group, i, pid, planets_by_id, int(self.max_ticks)
                )

        self.target_xy = self._planet_trajectory(
            self.target_id,
            planets_by_id,
            initial_by_id,
            comet_pid_set,
            orbital_ids,
            comet_keyframes,
        )

        blocker_xy_list: list[np.ndarray] = []
        blocker_r_list: list[float] = []
        for pid in blocker_ids:
            xy = self._planet_trajectory(
                pid,
                planets_by_id,
                initial_by_id,
                comet_pid_set,
                orbital_ids,
                comet_keyframes,
            )
            r = float(planets_by_id[pid][4])
            if not _trajectory_overlaps_board(xy, r):
                continue
            blocker_xy_list.append(xy)
            blocker_r_list.append(r)

        if blocker_xy_list:
            self.blocker_xy = np.stack(blocker_xy_list, axis=0)
            self.blocker_r = np.array(blocker_r_list, dtype=np.float64)
        else:
            self.blocker_xy = np.empty((0, len(self.tau), 2), dtype=np.float64)
            self.blocker_r = np.empty(0, dtype=np.float64)

    def _planet_trajectory(
        self,
        pid: int,
        planets_by_id: dict[int, PlanetRow],
        initial_by_id: dict[int, PlanetRow],
        comet_pid_set: set[int],
        orbital_ids: set[int],
        comet_keyframes: dict[int, np.ndarray],
    ) -> np.ndarray:
        if pid in comet_pid_set:
            return _sample_comet_on_tau(comet_keyframes[pid], self.tau)
        if pid in orbital_ids:
            initial_p = initial_by_id[pid]
            dx = float(initial_p[2]) - CENTER
            dy = float(initial_p[3]) - CENTER
            r = math.hypot(dx, dy)
            alpha = math.atan2(dy, dx)
            ang = alpha + self.omega * (self.start_step + self.tau)
            return np.column_stack(
                (CENTER + r * np.cos(ang), CENTER + r * np.sin(ang))
            )
        p = planets_by_id[pid]
        x, y = float(p[2]), float(p[3])
        return np.broadcast_to(np.array([[x, y]], dtype=np.float64), (len(self.tau), 2)).copy()

    def _fleet_grid(self, theta: float, tau: np.ndarray) -> np.ndarray:
        c, sn = math.cos(theta), math.sin(theta)
        s = self.s0 + self.v * tau
        return np.column_stack((self.cx + s * c, self.cy + s * sn))

    def _contact_gap_scalar(self, theta: float, tau: float) -> float:
        c, sn = math.cos(theta), math.sin(theta)
        s = self.s0 + self.v * tau
        fx = self.cx + s * c
        fy = self.cy + s * sn
        px = float(np.interp(tau, self.tau, self.target_xy[:, 0]))
        py = float(np.interp(tau, self.tau, self.target_xy[:, 1]))
        return math.hypot(fx - px, fy - py) - self.r_target

    def time_to_target_contact(self, theta: float) -> float:
        c, sn = math.cos(theta), math.sin(theta)
        s = self.s0 + self.v * self.tau
        fx = self.cx + s * c
        fy = self.cy + s * sn
        gap = np.hypot(fx - self.target_xy[:, 0], fy - self.target_xy[:, 1]) - self.r_target

        if gap[0] <= 0.0:
            return 0.0

        inside = gap <= 0.0
        if not inside.any():
            return _INF

        idx = int(np.argmax(inside))
        lo = float(self.tau[idx - 1]) if idx > 0 else 0.0
        hi = float(self.tau[idx])
        f = lambda tau: self._contact_gap_scalar(theta, tau)
        return _bisect_first_nonpositive(lo, hi, f)

    def _reach_gap_scalar(self, tau: float) -> float:
        px = float(np.interp(tau, self.tau, self.target_xy[:, 0]))
        py = float(np.interp(tau, self.tau, self.target_xy[:, 1]))
        dist = math.hypot(px - self.cx, py - self.cy)
        return dist - (self.s0 + self.v * tau) - self.r_target

    def lead_theta(self) -> float:
        dist = np.hypot(
            self.target_xy[:, 0] - self.cx, self.target_xy[:, 1] - self.cy
        )
        reach_gap = dist - (self.s0 + self.v * self.tau) - self.r_target

        px0, py0 = self.target_xy[0, 0], self.target_xy[0, 1]
        if reach_gap[0] <= 0.0:
            return math.atan2(py0 - self.cy, px0 - self.cx)

        inside = reach_gap <= 0.0
        if not inside.any():
            return math.atan2(py0 - self.cy, px0 - self.cx)

        idx = int(np.argmax(inside))
        lo = float(self.tau[idx - 1]) if idx > 0 else 0.0
        hi = float(self.tau[idx])
        tau_lead = _bisect_first_nonpositive(lo, hi, self._reach_gap_scalar)
        px = float(np.interp(tau_lead, self.tau, self.target_xy[:, 0]))
        py = float(np.interp(tau_lead, self.tau, self.target_xy[:, 1]))
        return math.atan2(py - self.cy, px - self.cx)

    def _active_blockers(self, theta: float, t_hit: float) -> np.ndarray:
        if self.blocker_xy.shape[0] == 0:
            return np.zeros(0, dtype=bool)

        c, sn = math.cos(theta), math.sin(theta)
        u = np.array([c, sn], dtype=np.float64)
        origin = np.array([self.cx, self.cy], dtype=np.float64)
        rel = self.blocker_xy - origin
        along = rel @ u
        perp2 = np.sum(rel * rel, axis=2) - along * along
        d_min = np.sqrt(np.maximum(perp2, 0.0)).min(axis=1)
        s_max = self.s0 + self.v * min(t_hit, self.max_ticks)
        margin = self.dt * self.v
        return (
            (d_min < self.blocker_r + margin)
            & (along.min(axis=1) < s_max)
            & (along.max(axis=1) > -margin)
        )

    def _earliest_sun_tau(self, theta: float, t_limit: float) -> float:
        c, sn = math.cos(theta), math.sin(theta)
        s_end = self.s0 + self.v * max(t_limit, 0.0)
        ax = self.cx + self.s0 * c
        ay = self.cy + self.s0 * sn
        bx = self.cx + s_end * c
        by = self.cy + s_end * sn
        if _point_segment_distance(CENTER, CENTER, ax, ay, bx, by) >= self.sun_radius:
            return _INF

        def dist_at_s(s: float) -> float:
            x = self.cx + s * c
            y = self.cy + s * sn
            return math.hypot(x - CENTER, y - CENTER)

        if dist_at_s(self.s0) <= self.sun_radius:
            return 0.0

        s_lo, s_hi = self.s0, s_end
        for _ in range(_BISECT_ITERS):
            if s_hi - s_lo < _TAU_EPS * max(self.v, 1e-9):
                break
            mid = 0.5 * (s_lo + s_hi)
            if dist_at_s(mid) <= self.sun_radius:
                s_hi = mid
            else:
                s_lo = mid
        if self.v <= 0.0:
            return 0.0
        return max(0.0, (s_hi - self.s0) / self.v)

    def evaluate(self, theta: float) -> tuple[float, bool]:
        t_hit = self.time_to_target_contact(theta)
        if not math.isfinite(t_hit) or t_hit == _INF:
            return _INF, False

        k = min(int(np.searchsorted(self.tau, t_hit)), len(self.tau) - 1)
        tau_slice = self.tau[: k + 1]
        fleet = self._fleet_grid(theta, tau_slice)

        oob = (
            (fleet[:, 0] < 0.0)
            | (fleet[:, 0] > BOARD_SIZE)
            | (fleet[:, 1] < 0.0)
            | (fleet[:, 1] > BOARD_SIZE)
        )
        t_oob = float(tau_slice[int(np.argmax(oob))]) if oob.any() else _INF

        t_sun = self._earliest_sun_tau(theta, min(t_hit, self.max_ticks))

        active = self._active_blockers(theta, t_hit)
        t_block = _INF
        if active.any():
            block = self.blocker_xy[active, : k + 1, :]
            diff = fleet[np.newaxis, :, :] - block
            dist = np.linalg.norm(diff, axis=2) - self.blocker_r[active, np.newaxis]
            viol = (dist <= 0.0).any(axis=0)
            if viol.any():
                t_block = float(tau_slice[int(np.flatnonzero(viol)[0])])

        t_limit = min(t_block, t_sun, t_oob)
        feasible = t_hit < t_limit - _TAU_EPS
        return t_hit, feasible


def _golden_section_min(lo: float, hi: float, f, iters: int = 12) -> float:
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
    candidates = [theta0, theta0 + span, theta0 - span, naive]

    best_theta: float | None = None
    best_t = _INF
    best_feasible_theta: float | None = None
    best_feasible_t = _INF

    for a in candidates:
        a = _normalize_angle(a)
        t_hit, ok = model.evaluate(a)
        if ok and t_hit < best_feasible_t:
            best_feasible_t, best_feasible_theta = t_hit, a
        if math.isfinite(t_hit) and t_hit < best_t:
            best_t, best_theta = t_hit, a

    seed = best_feasible_theta if best_feasible_theta is not None else best_theta
    if seed is None:
        seed = naive

    lo = seed - span
    hi = seed + span
    refined = _normalize_angle(_golden_section_min(lo, hi, cost))
    t_ref, ok_ref = model.evaluate(refined)
    if ok_ref and t_ref < best_feasible_t:
        best_feasible_t, best_feasible_theta = t_ref, refined

    if best_feasible_theta is not None:
        return best_feasible_theta, True

    if best_theta is not None and math.isfinite(best_t):
        return best_theta, False

    return naive, False
