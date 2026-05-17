"""
Orbit Wars - Nearest Planet Sniper Agent

A simple agent that captures the nearest unowned planet when it has
enough ships to guarantee the takeover.

Strategy:
  For each planet we own, find the closest planet we don't own.
  If we have more ships than the target's garrison, send exactly
  enough to capture it (garrison + 1). Otherwise, wait and accumulate.

Key concepts demonstrated:
  - Parsing the observation (planets, player ID, dynamics fields)
  - Fleet heading from continuous_intercept.intercept_angle_for_target
  - Sending moves as [from_planet_id, angle, num_ships]
"""

import math
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from continuous_intercept import intercept_angle_for_target
#from orbit_dynamics import intercept_angle_for_target


def _obs_get(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def agent(obs):
    moves = []
    player = int(_obs_get(obs, "player", 0))
    raw_planets = _obs_get(obs, "planets", []) or []
    planets_rows = list(raw_planets)
    initial = list(_obs_get(obs, "initial_planets", []) or [])
    omega = float(_obs_get(obs, "angular_velocity", 0.0))
    step = int(_obs_get(obs, "step", 0))
    comets = list(_obs_get(obs, "comets") or [])
    cids = list(_obs_get(obs, "comet_planet_ids") or [])
    cfg = _obs_get(obs, "configuration")

    by_id = {int(p[0]): p for p in planets_rows}

    # Parse into named tuples for readable field access:
    #   Planet(id, owner, x, y, radius, ships, production)
    #   owner == -1 means neutral, 0-3 are player IDs
    planets = [Planet(*p) for p in raw_planets]
    my_planets = [p for p in planets if p.owner == player]
    targets = [p for p in planets if p.owner != player]

    if not targets:
        return moves

    for mine in my_planets:
        # Find the nearest planet we don't own
        nearest = None
        min_dist = float("inf")
        for t in targets:
            dist = math.sqrt((mine.x - t.x) ** 2 + (mine.y - t.y) ** 2)
            if dist < min_dist:
                min_dist = dist
                nearest = t

        if nearest is None:
            continue

        # We need to send more ships than the target has to capture it.
        # Exactly target_ships + 1 guarantees the takeover.
        ships_needed = nearest.ships + 1

        # Only launch if we can afford it — otherwise keep accumulating
        if mine.ships >= ships_needed:
            mine_row = by_id[int(mine.id)]
            nearest_row = by_id[int(nearest.id)]
            angle, _feasible = intercept_angle_for_target(
                mine_row,
                nearest_row,
                ships_needed,
                planets_rows,
                initial,
                omega,
                step,
                comets,
                cids,
                cfg,
            )
            if angle is None:
                angle = math.atan2(nearest.y - mine.y, nearest.x - mine.x)
            moves.append([mine.id, angle, ships_needed])

    return moves
