"""Observation encoding and action masks for Orbit Wars."""

from __future__ import annotations

from typing import Any

import numpy as np

MAX_PLANETS = 40
PLANET_FEAT_DIM = 11
GLOBAL_FEAT_DIM = 5
OBS_DIM = MAX_PLANETS * PLANET_FEAT_DIM + GLOBAL_FEAT_DIM
NUM_SEND_MODES = 4


def obs_get(obs: Any, key: str, default: Any = None) -> Any:
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _slot_assignment(planets_rows: list) -> tuple[np.ndarray, np.ndarray]:
    """Map slot index -> planet id; -1 for padded slots."""
    ids = sorted(int(p[0]) for p in planets_rows)[:MAX_PLANETS]
    slot_ids = np.full(MAX_PLANETS, -1, dtype=np.int64)
    valid = np.zeros(MAX_PLANETS, dtype=bool)
    for i, pid in enumerate(ids):
        slot_ids[i] = pid
        valid[i] = True
    return slot_ids, valid


def encode_observation(obs: Any) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Flatten planets into fixed slots (sorted by planet id) plus global features.

    Returns:
        obs_vec: float32 (OBS_DIM,)
        info: source_mask, target_mask, slot_planet_ids, planet_valid
    """
    player = int(obs_get(obs, "player", 0))
    planets_rows = list(obs_get(obs, "planets", []) or [])
    step = int(obs_get(obs, "step", 0))
    omega = float(obs_get(obs, "angular_velocity", 0.0))

    by_id = {int(p[0]): p for p in planets_rows}
    slot_ids, planet_valid = _slot_assignment(planets_rows)

    feats = np.zeros((MAX_PLANETS, PLANET_FEAT_DIM), dtype=np.float32)
    source_mask = np.zeros(MAX_PLANETS, dtype=bool)
    target_mask = np.zeros(MAX_PLANETS, dtype=bool)

    n_mine = 0
    n_enemy = 0

    for slot in range(MAX_PLANETS):
        if not planet_valid[slot]:
            continue
        pid = int(slot_ids[slot])
        p = by_id.get(pid)
        if p is None:
            planet_valid[slot] = False
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

        feats[slot] = [
            x,
            y,
            1.0 if is_mine else 0.0,
            1.0 if is_enemy else 0.0,
            1.0 if is_neutral else 0.0,
            ships,
            radius,
            production,
            float(is_mine),
            float(is_enemy),
            1.0,
        ]

        if is_mine and int(p[5]) > 0:
            source_mask[slot] = True
        if owner != player:
            target_mask[slot] = True

    global_feat = np.array(
        [
            step / 500.0,
            omega,
            n_mine / MAX_PLANETS,
            n_enemy / MAX_PLANETS,
            float(player),
        ],
        dtype=np.float32,
    )

    obs_vec = np.concatenate([feats.ravel(), global_feat]).astype(np.float32)
    assert obs_vec.shape == (OBS_DIM,)

    info = {
        "source_mask": source_mask,
        "target_mask": target_mask,
        "planet_valid": planet_valid,
        "slot_planet_ids": slot_ids,
    }
    return obs_vec, info
