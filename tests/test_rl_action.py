"""Tests for RL action decoding."""

from __future__ import annotations

from kaggle_environments import make

from rl.action import decode_action, ships_for_mode
from rl.encoding import encode_observation


def test_ships_for_mode():
    assert ships_for_mode(0, 50, 10) == 0
    assert ships_for_mode(1, 50, 10) == 11
    assert ships_for_mode(2, 50, 10) == 25
    assert ships_for_mode(3, 50, 10) == 50


def test_decode_action_valid_move():
    env = make("orbit_wars", configuration={"seed": 7, "episodeSteps": 50}, debug=True)

    def noop(obs):
        return []

    env.run([noop, noop])
    obs = env.steps[1][0].observation
    _, info = encode_observation(obs)

    src = int(info["source_mask"].argmax())
    tgt = int(info["target_mask"].argmax())
    moves = decode_action(obs, (src, tgt, 1), info["slot_planet_ids"])
    assert len(moves) == 1
    assert len(moves[0]) == 3
    from_id, angle, ships = moves[0]
    assert isinstance(from_id, int)
    assert isinstance(angle, float)
    assert ships >= 1
