"""Dynamics regression tests (run with: conda run -n crypto pytest tests/test_orbit_dynamics.py -v)."""

from __future__ import annotations

import math

import pytest

from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import interpreter

from orbit_dynamics import (
    fleet_speed,
    predict_planet_xy_after_tick,
    rollout_first_event_code,
)


@pytest.fixture
def cfg():
    return {"shipSpeed": 6.0, "sunRadius": 10.0, "episodeSteps": 500, "cometSpeed": 4.0}


def test_fleet_speed_matches_engine_formula(cfg):
    assert fleet_speed(1, cfg) == pytest.approx(1.0)
    assert fleet_speed(1000, cfg) == pytest.approx(6.0)


def test_predict_end_xy_matches_next_step(cfg):
    env = make("orbit_wars", configuration={"seed": 42, "episodeSteps": 30}, debug=True)

    def noop(obs):
        return []

    env.run([noop, noop])
    assert len(env.steps) >= 3
    obs0 = env.steps[1][0].observation
    obs1 = env.steps[2][0].observation
    planets0 = list(obs0.planets)
    initial = list(obs0.initial_planets)
    omega = float(obs0.angular_velocity)
    comets = list(getattr(obs0, "comets", []) or [])
    cids = list(getattr(obs0, "comet_planet_ids", []) or [])
    st = int(getattr(obs0, "step", 0))

    for p in planets0:
        ex, ey = predict_planet_xy_after_tick(p, planets0, initial, omega, st, comets, cids)
        p1 = next(x for x in obs1.planets if x[0] == p[0])
        assert ex == pytest.approx(float(p1[2]))
        assert ey == pytest.approx(float(p1[3]))


def test_rollout_tunnel_rotating_planet(cfg):
    """Same geometry as orbit_wars test_fleet_does_not_tunnel_through_rotating_planet."""
    from types import SimpleNamespace

    planets = [[0, -1, 50.0, 52.0, 1.0, 10, 1]]
    fleets = [[0, 0, 49.0, 50.0, 0.0, 1, 1000]]
    state = [
        SimpleNamespace(
            observation=SimpleNamespace(
                player=0,
                step=1,
                planets=[p[:] for p in planets],
                fleets=[f[:] for f in fleets],
                next_fleet_id=1,
                angular_velocity=math.pi,
                initial_planets=[p[:] for p in planets],
                comets=[],
                comet_planet_ids=[],
            ),
            action=[],
            status="ACTIVE",
            reward=0,
        ),
        SimpleNamespace(
            observation=SimpleNamespace(player=1),
            action=[],
            status="ACTIVE",
            reward=0,
        ),
    ]
    env = SimpleNamespace(
        configuration=SimpleNamespace(shipSpeed=2, episodeSteps=500, cometSpeed=4),
        done=False,
    )
    new_state = interpreter(state, env)
    assert len(new_state[0].observation.fleets) == 0

    from_p = [99, 0, 48.0, 50.0, 1.0, 1000, 1]
    obs = {
        "planets": [planets[0][:]],
        "initial_planets": [planets[0][:]],
        "angular_velocity": math.pi,
        "comets": [],
        "comet_planet_ids": [],
        "step": 1,
    }
    code, extra = rollout_first_event_code(
        from_p,
        0.0,
        1000,
        list(obs["planets"]),
        list(obs["initial_planets"]),
        float(obs["angular_velocity"]),
        int(obs["step"]),
        [],
        [],
        target_id=1,
        configuration=env.configuration,
        max_ticks=5,
    )
    assert code == 2
    assert extra == 0
