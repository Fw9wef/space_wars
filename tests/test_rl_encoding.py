"""Tests for RL observation encoding."""

from __future__ import annotations

import numpy as np
from kaggle_environments import make

from rl.encoding import MAX_PLANETS, OBS_DIM, encode_observation


def test_encode_observation_shape():
    env = make("orbit_wars", configuration={"seed": 42, "episodeSteps": 30}, debug=True)

    def noop(obs):
        return []

    env.run([noop, noop])
    obs = env.steps[1][0].observation
    vec, info = encode_observation(obs)

    assert vec.shape == (OBS_DIM,)
    assert vec.dtype == np.float32
    assert info["source_mask"].shape == (MAX_PLANETS,)
    assert info["target_mask"].shape == (MAX_PLANETS,)
    assert info["slot_planet_ids"].shape == (MAX_PLANETS,)
    assert info["planet_valid"].sum() == len(obs.planets)
