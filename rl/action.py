"""Discrete action decoding -> orbit_wars moves."""

from __future__ import annotations

import math
from typing import Any

from continuous_intercept import intercept_angle_for_target

from rl.encoding import NUM_SEND_MODES, obs_get

ActionTuple = tuple[int, int, int]


def noop_action() -> ActionTuple:
    return (0, 0, 0)


def ships_for_mode(send_mode: int, mine_ships: int, target_ships: int) -> int:
    if send_mode <= 0:
        return 0
    if send_mode == 1:
        return min(mine_ships, target_ships + 1)
    if send_mode == 2:
        return max(1, mine_ships // 2)
    return mine_ships


def decode_action(
    obs: Any,
    action: ActionTuple,
    slot_planet_ids: Any,
    *,
    force_noop: bool = False,
) -> list[list[float | int]]:
    """Decode (source_slot, target_slot, send_mode) into [[from_id, angle, ships]]."""
    if force_noop:
        return []

    source_slot, target_slot, send_mode = (int(action[0]), int(action[1]), int(action[2]))
    if send_mode <= 0 or send_mode >= NUM_SEND_MODES:
        return []

    slot_ids = list(slot_planet_ids)
    if source_slot < 0 or source_slot >= len(slot_ids):
        return []
    if target_slot < 0 or target_slot >= len(slot_ids):
        return []

    source_id = int(slot_ids[source_slot])
    target_id = int(slot_ids[target_slot])
    if source_id < 0 or target_id < 0:
        return []

    player = int(obs_get(obs, "player", 0))
    planets_rows = list(obs_get(obs, "planets", []) or [])
    by_id = {int(p[0]): p for p in planets_rows}

    mine_row = by_id.get(source_id)
    target_row = by_id.get(target_id)
    if mine_row is None or target_row is None:
        return []

    if int(mine_row[1]) != player:
        return []
    if int(target_row[1]) == player:
        return []

    mine_ships = int(mine_row[5])
    target_ships = int(target_row[5])
    num_ships = ships_for_mode(send_mode, mine_ships, target_ships)
    if num_ships <= 0 or num_ships > mine_ships:
        return []

    initial = list(obs_get(obs, "initial_planets", []) or [])
    omega = float(obs_get(obs, "angular_velocity", 0.0))
    step = int(obs_get(obs, "step", 0))
    comets = list(obs_get(obs, "comets") or [])
    cids = list(obs_get(obs, "comet_planet_ids") or [])
    cfg = obs_get(obs, "configuration")

    angle, _feasible = intercept_angle_for_target(
        mine_row,
        target_row,
        num_ships,
        planets_rows,
        initial,
        omega,
        step,
        comets,
        cids,
        cfg,
    )
    if angle is None:
        angle = math.atan2(
            float(target_row[3]) - float(mine_row[3]),
            float(target_row[2]) - float(mine_row[2]),
        )

    return [[source_id, float(angle), int(num_ships)]]
