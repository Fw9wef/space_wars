"""
Orbit Wars dynamics helpers aligned with kaggle_environments orbit_wars.py.

Interpreter phase order (see site-packages/.../orbit_wars.py `interpreter`):
  comet expiry → comet spawn → fleet launch → production → precompute planet
  paths (comet path_index += 1 here) → fleet movement (swept_pair_hit vs each
  planet in list order, then OOB, then sun) → apply planet positions → combat.

Planet rows: [id, owner, x, y, radius, ships, production] — same as engine lists.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Iterable, Sequence

from kaggle_environments.envs.orbit_wars.orbit_wars import (
    BOARD_SIZE,
    CENTER,
    ROTATION_RADIUS_LIMIT,
    SUN_RADIUS,
    point_to_segment_distance,
    swept_pair_hit,
)

# Re-export for callers
__all__ = [
    "BOARD_SIZE",
    "CENTER",
    "ROTATION_RADIUS_LIMIT",
    "SUN_RADIUS",
    "fleet_speed",
    "fleet_spawn_xy",
    "trajectory_hits_sun",
    "filter_safe_angles",
    "sun_safe_mask_angles",
    "intercept_angle_for_target",
    "rollout_first_event",
    "rollout_first_event_code",
    "reach_indicator_for_target",
    "reach_indicators_from_planet",
    "predict_planet_xy_after_tick",
    "build_planet_paths",
]

ObsDict = dict[str, Any]
PlanetRow = list
Configuration = dict[str, float] | Any


def _get_cfg(configuration: Configuration, key: str, default: float) -> float:
    if isinstance(configuration, dict):
        return float(configuration.get(key, default))
    return float(getattr(configuration, key, default))


def fleet_speed(ships: int, configuration: Configuration) -> float:
    """Same formula as orbit_wars interpreter (movement block)."""
    ships = max(1, int(ships))
    max_speed = _get_cfg(configuration, "shipSpeed", 6.0)
    speed = 1.0 + (max_speed - 1.0) * (math.log(ships) / math.log(1000.0)) ** 1.5
    return min(speed, max_speed)


def fleet_spawn_xy(planet: PlanetRow, angle: float) -> tuple[float, float]:
    """Fleet spawn just outside planet radius (engine uses +0.1)."""
    r = float(planet[4])
    x = float(planet[2]) + math.cos(angle) * (r + 0.1)
    y = float(planet[3]) + math.sin(angle) * (r + 0.1)
    return x, y


def build_planet_paths(
    planets: Sequence[PlanetRow],
    initial_planets: Sequence[PlanetRow],
    angular_velocity: float,
    step: int,
    comets: Iterable[dict[str, Any]],
    comet_planet_ids: Sequence[int],
) -> dict[int, tuple[tuple[float, float], tuple[float, float], bool]]:
    """
    Mirror orbit_wars interpreter: precompute each planet's movement chord for
    this tick. `step` must match observation.step when the tick is processed.
    """
    comet_pid_set = set(comet_planet_ids)
    initial_by_id = {p[0]: p for p in initial_planets}
    planet_paths: dict[int, tuple[tuple[float, float], tuple[float, float], bool]] = {}

    for planet in planets:
        if planet[0] in comet_pid_set:
            continue
        old_pos = (float(planet[2]), float(planet[3]))
        new_pos = old_pos
        initial_p = initial_by_id.get(planet[0])
        if initial_p is not None:
            dx = float(initial_p[2]) - CENTER
            dy = float(initial_p[3]) - CENTER
            r = math.hypot(dx, dy)
            if r + float(planet[4]) < ROTATION_RADIUS_LIMIT:
                initial_angle = math.atan2(dy, dx)
                current_angle = initial_angle + float(angular_velocity) * float(step)
                new_pos = (
                    CENTER + r * math.cos(current_angle),
                    CENTER + r * math.sin(current_angle),
                )
        planet_paths[planet[0]] = (old_pos, new_pos, True)

    for group in comets:
        idx = int(group["path_index"]) + 1
        for i, pid in enumerate(group["planet_ids"]):
            planet = next((p for p in planets if p[0] == pid), None)
            if planet is None:
                continue
            p_path = group["paths"][i]
            old_pos = (float(planet[2]), float(planet[3]))
            if idx >= len(p_path):
                planet_paths[pid] = (old_pos, old_pos, True)
            else:
                new_pos = (float(p_path[idx][0]), float(p_path[idx][1]))
                check = old_pos[0] >= 0.0
                planet_paths[pid] = (old_pos, new_pos, check)

    return planet_paths


def predict_planet_xy_after_tick(
    planet: PlanetRow,
    all_planets: Sequence[PlanetRow],
    initial_planets: Sequence[PlanetRow],
    angular_velocity: float,
    step: int,
    comets: Iterable[dict[str, Any]],
    comet_planet_ids: Sequence[int],
) -> tuple[float, float]:
    """End-of-tick center position for one planet (matches interpreter apply step)."""
    paths = build_planet_paths(
        list(all_planets), initial_planets, angular_velocity, step, comets, comet_planet_ids
    )
    _old, new_pos, _c = paths[planet[0]]
    return new_pos


def _fleet_segment_hits_sun(
    old_pos: tuple[float, float], new_pos: tuple[float, float], sun_radius: float
) -> bool:
    return point_to_segment_distance((CENTER, CENTER), old_pos, new_pos) < sun_radius


def _oob(pos: tuple[float, float]) -> bool:
    x, y = pos
    return not (0.0 <= x <= BOARD_SIZE and 0.0 <= y <= BOARD_SIZE)


def _first_fleet_collision(
    old_pos: tuple[float, float],
    new_pos: tuple[float, float],
    planets: Sequence[PlanetRow],
    planet_paths: dict[int, tuple[tuple[float, float], tuple[float, float], bool]],
) -> int | None:
    """First planet id whose swept chord hits the fleet segment (engine order)."""
    for planet in planets:
        path = planet_paths.get(planet[0])
        if path is None or not path[2]:
            continue
        p_old, p_new, _ = path
        if swept_pair_hit(old_pos, new_pos, p_old, p_new, float(planet[4])):
            return int(planet[0])
    return None


def trajectory_hits_sun(
    launch_xy: tuple[float, float],
    angle: float,
    fleet_ships: int,
    configuration: Configuration,
    max_steps: int = 500,
    *,
    sun_radius: float | None = None,
) -> tuple[bool, int | None]:
    """
    Straight-line fleet ignoring planets — only OOB and sun (segment vs sun each step).
    Returns (hit_sun, step_index_if_hit) where step_index is 0-based movement step.
    """
    sr = float(sun_radius if sun_radius is not None else _get_cfg(configuration, "sunRadius", SUN_RADIUS))
    speed = fleet_speed(fleet_ships, configuration)
    x, y = float(launch_xy[0]), float(launch_xy[1])
    ca, sa = math.cos(angle), math.sin(angle)
    for t in range(max_steps):
        old = (x, y)
        x += ca * speed
        y += sa * speed
        new = (x, y)
        if _fleet_segment_hits_sun(old, new, sr):
            return True, t
        if _oob(new):
            return False, None
    return False, None


def filter_safe_angles(
    launch_xy: tuple[float, float],
    fleet_ships: int,
    configuration: Configuration,
    angle_grid: Sequence[float],
    *,
    max_steps: int = 500,
    sun_radius: float | None = None,
) -> list[bool]:
    """Sun safety for a fixed spawn point with varying headings (rarely correct for Orbit Wars)."""
    out: list[bool] = []
    for a in angle_grid:
        hit, _ = trajectory_hits_sun(
            launch_xy,
            float(a),
            fleet_ships,
            configuration,
            max_steps=max_steps,
            sun_radius=sun_radius,
        )
        out.append(not hit)
    return out


def sun_safe_mask_angles(
    from_planet: PlanetRow,
    fleet_ships: int,
    configuration: Configuration,
    angle_grid: Sequence[float],
    *,
    max_steps: int = 500,
) -> list[bool]:
    """Per-angle sun safety using correct spawn point for each heading."""
    out: list[bool] = []
    for a in angle_grid:
        lx, ly = fleet_spawn_xy(from_planet, float(a))
        hit, _ = trajectory_hits_sun(
            (lx, ly), float(a), fleet_ships, configuration, max_steps=max_steps
        )
        out.append(not hit)
    return out


def _rollout_one_move(
    fleet_xy: tuple[float, float],
    angle: float,
    fleet_ships: int,
    planets: list[PlanetRow],
    planet_paths: dict[int, tuple[tuple[float, float], tuple[float, float], bool]],
    configuration: Configuration,
    target_id: int,
    sun_radius: float,
) -> tuple[str, int | None]:
    """
    One fleet movement step (same ordering as interpreter).
    Returns (event, detail_id): ('target', j), ('planet', pid), ('sun', None), ('oob', None), ('none', None)
    """
    speed = fleet_speed(fleet_ships, configuration)
    old_pos = (float(fleet_xy[0]), float(fleet_xy[1]))
    x = old_pos[0] + math.cos(angle) * speed
    y = old_pos[1] + math.sin(angle) * speed
    new_pos = (x, y)

    hit_planet = _first_fleet_collision(old_pos, new_pos, planets, planet_paths)
    if hit_planet is not None:
        if hit_planet == target_id:
            return "target", target_id
        return "planet", hit_planet

    if _oob(new_pos):
        return "oob", None

    if _fleet_segment_hits_sun(old_pos, new_pos, sun_radius):
        return "sun", None

    return "none", None


def _advance_planets_inplace(
    planets: list[PlanetRow],
    planet_paths: dict[int, tuple[tuple[float, float], tuple[float, float], bool]],
) -> None:
    for p in planets:
        path = planet_paths.get(p[0])
        if path is not None:
            nx, ny = path[1]
            p[2], p[3] = nx, ny


def _bump_comet_path_indices(comets: list[dict[str, Any]]) -> None:
    """Mirror interpreter: comet path_index is incremented once per tick during path build."""
    for group in comets:
        group["path_index"] = int(group["path_index"]) + 1


def rollout_first_event(
    from_planet: PlanetRow,
    angle: float,
    fleet_ships: int,
    planets: list[PlanetRow],
    initial_planets: list[PlanetRow],
    angular_velocity: float,
    start_step: int,
    comets: list[dict[str, Any]],
    comet_planet_ids: list[int],
    target_id: int,
    configuration: Configuration,
    max_ticks: int = 400,
    *,
    fleet_xy: tuple[float, float] | None = None,
) -> tuple[int, int | None, int]:
    """
    Discrete rollout aligned with engine movement + planet advance each tick.

    Returns (code, extra, ticks):
      0 — first contact is target disk (extra target_id)
      1 — sun first
      2 — other planet first (extra planet_id)
      3 — horizon exhausted (no target contact)
      4 — out of bounds first (extra None)
      ticks — 0-based movement step index when event occurred, or max_ticks if code==3
    """
    sun_radius = _get_cfg(configuration, "sunRadius", SUN_RADIUS)
    planets = copy.deepcopy(planets)
    comets = copy.deepcopy(comets)
    comet_ids = list(comet_planet_ids)
    step = int(start_step)
    if fleet_xy is None:
        fx, fy = fleet_spawn_xy(from_planet, angle)
    else:
        fx, fy = float(fleet_xy[0]), float(fleet_xy[1])

    for tick in range(max_ticks):
        paths = build_planet_paths(planets, initial_planets, angular_velocity, step, comets, comet_ids)
        ev, detail = _rollout_one_move(
            (fx, fy), angle, fleet_ships, planets, paths, configuration, target_id, sun_radius
        )
        if ev == "target":
            return 0, target_id, tick
        if ev == "planet":
            return 2, int(detail) if detail is not None else None, tick
        if ev == "sun":
            return 1, None, tick
        if ev == "oob":
            return 4, None, tick

        speed = fleet_speed(fleet_ships, configuration)
        fx += math.cos(angle) * speed
        fy += math.sin(angle) * speed

        _advance_planets_inplace(planets, paths)
        _bump_comet_path_indices(comets)
        step += 1

    return 3, None, max_ticks


def rollout_first_event_code(
    from_planet: PlanetRow,
    angle: float,
    fleet_ships: int,
    planets: list[PlanetRow],
    initial_planets: list[PlanetRow],
    angular_velocity: float,
    start_step: int,
    comets: list[dict[str, Any]],
    comet_planet_ids: list[int],
    target_id: int,
    configuration: Configuration,
    max_ticks: int = 400,
) -> tuple[int, int | None]:
    code, extra, _ticks = rollout_first_event(
        from_planet,
        angle,
        fleet_ships,
        planets,
        initial_planets,
        angular_velocity,
        start_step,
        comets,
        comet_planet_ids,
        target_id,
        configuration,
        max_ticks,
    )
    return code, extra


def _aim_naive(from_planet: PlanetRow, target: PlanetRow) -> float:
    """Direction from source planet center toward target center (static snap)."""
    return math.atan2(float(target[3]) - float(from_planet[3]), float(target[2]) - float(from_planet[2]))


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
    coarse: int = 48,
    max_ticks: int = 400,
) -> tuple[float | None, bool]:
    """
    Search heading that reaches target first (code 0) within max_ticks.
    Returns (angle, feasible). If infeasible, returns (best_naive_angle_or_None, False).
    """
    tid = int(target[0])
    pl = copy.deepcopy(list(planets))
    cm = copy.deepcopy(comets)
    cids = list(comet_planet_ids)

    def score(angle: float) -> float:
        code, _ = rollout_first_event_code(
            from_planet,
            angle,
            fleet_ships,
            copy.deepcopy(pl),
            copy.deepcopy(list(initial_planets)),
            angular_velocity,
            start_step,
            copy.deepcopy(cm),
            cids,
            tid,
            configuration,
            max_ticks=max_ticks,
        )
        if code == 0:
            return 0.0
        if code == 1:
            return 1e3
        if code == 2:
            return 2e3
        if code == 4:
            return 3e3
        return 5e3

    best_a = None
    best_s = float("inf")
    for k in range(coarse):
        a = 2 * math.pi * k / coarse
        s = score(a)
        if s < best_s:
            best_s, best_a = s, a

    if best_a is None:
        return None, False

    # local refinement (golden window around best coarse bin)
    delta = (2 * math.pi) / coarse
    lo = best_a - delta
    hi = best_a + delta
    for _ in range(10):
        mid1 = lo + (hi - lo) * 0.38
        mid2 = lo + (hi - lo) * 0.62
        if score(mid1) < score(mid2):
            hi = mid2
        else:
            lo = mid1

    refined = 0.5 * (lo + hi)
    if score(refined) == 0.0:
        return refined, True

    naive = _aim_naive(from_planet, target)
    if score(naive) == 0.0:
        return naive, True

    return (naive if best_s >= 5e3 else best_a), False


def reach_indicator_for_target(
    from_planet: PlanetRow,
    target: PlanetRow,
    fleet_ships: int,
    obs: ObsDict,
    configuration: Configuration,
    *,
    angle: float | None = None,
    max_ticks: int = 400,
) -> tuple[int, bool, float]:
    """
    Returns (code, intercept_feasible, angle_used):
      0 reach target first, 1 sun, 2 other planet, 3 horizon, 4 oob
    If angle is None, uses intercept_angle_for_target.
    """
    planets = list(obs["planets"])
    initial = list(obs["initial_planets"])
    omega = float(obs["angular_velocity"])
    step = int(obs.get("step", 0))
    comets = list(obs.get("comets") or [])
    cids = list(obs.get("comet_planet_ids") or [])

    if angle is None:
        angle, feas = intercept_angle_for_target(
            from_planet,
            target,
            fleet_ships,
            planets,
            initial,
            omega,
            step,
            comets,
            cids,
            configuration,
            max_ticks=max_ticks,
        )
        if angle is None:
            return 3, False, float("nan")
    else:
        feas = (
            rollout_first_event_code(
                from_planet,
                angle,
                fleet_ships,
                copy.deepcopy(planets),
                copy.deepcopy(initial),
                omega,
                step,
                copy.deepcopy(comets),
                cids,
                int(target[0]),
                configuration,
                max_ticks=max_ticks,
            )[0]
            == 0
        )

    code, _ = rollout_first_event_code(
        from_planet,
        float(angle),
        fleet_ships,
        copy.deepcopy(planets),
        copy.deepcopy(initial),
        omega,
        step,
        copy.deepcopy(comets),
        cids,
        int(target[0]),
        configuration,
        max_ticks=max_ticks,
    )
    return code, feas, float(angle)


def reach_indicators_from_planet(
    from_planet: PlanetRow,
    fleet_ships: int,
    obs: ObsDict,
    configuration: Configuration,
    *,
    max_ticks: int = 400,
) -> tuple[dict[int, int], dict[int, bool], dict[int, float]]:
    """
    For each other planet id as target, compute reach code {0,1,2,3,4},
    intercept_feasible flag, and chosen angle.
    """
    codes: dict[int, int] = {}
    feas: dict[int, bool] = {}
    angles: dict[int, float] = {}
    planets = obs["planets"]
    fid = int(from_planet[0])
    for p in planets:
        tid = int(p[0])
        if tid == fid:
            continue
        code, ok, ang = reach_indicator_for_target(
            from_planet, p, fleet_ships, obs, configuration, angle=None, max_ticks=max_ticks
        )
        codes[tid] = code
        feas[tid] = ok
        angles[tid] = ang
    return codes, feas, angles
