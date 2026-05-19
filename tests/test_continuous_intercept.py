"""Smoke tests for continuous_intercept."""

from __future__ import annotations

import math

from kaggle_environments import make

from continuous_intercept import _InterceptModel, intercept_angle_for_target


def _default_cfg() -> dict:
    return {"sunRadius": 40.0, "shipSpeed": 6.0}


def test_intercept_static_planets_smoke():
    from_p = [0, 1, 100.0, 200.0, 15.0, 50, 1]
    to_p = [1, 0, 300.0, 280.0, 20.0, 10, 1]
    planets = [from_p, to_p]
    initial = [list(p) for p in planets]

    angle, feasible = intercept_angle_for_target(
        from_p,
        to_p,
        10,
        planets,
        initial,
        angular_velocity=0.0,
        start_step=0,
        comets=[],
        comet_planet_ids=[],
        configuration=_default_cfg(),
        max_ticks=400,
        path_sample_dt=0.5,
    )

    assert angle is not None
    assert 0.0 <= angle < 2 * math.pi
    assert isinstance(feasible, bool)


def test_feasible_implies_finite_evaluate():
    env = make("orbit_wars", configuration={"seed": 11, "episodeSteps": 40}, debug=True)

    def noop(obs):
        return []

    env.run([noop, noop])
    obs = env.steps[1][0].observation
    cfg = env.configuration
    planets = list(obs.planets)
    initial = list(obs.initial_planets)
    player = int(obs.player)

    mine = next(p for p in planets if int(p[1]) == player and int(p[5]) > 5)
    target = next(
        p
        for p in planets
        if int(p[1]) != player and int(p[0]) != int(mine[0])
    )

    angle, feasible = intercept_angle_for_target(
        mine,
        target,
        min(5, int(mine[5])),
        planets,
        initial,
        float(obs.angular_velocity),
        int(obs.step),
        list(obs.comets or []),
        list(obs.comet_planet_ids or []),
        cfg,
    )
    assert angle is not None

    model = _InterceptModel(
        mine,
        target,
        min(5, int(mine[5])),
        planets,
        initial,
        float(obs.angular_velocity),
        int(obs.step),
        list(obs.comets or []),
        list(obs.comet_planet_ids or []),
        cfg,
        400,
        0.25,
    )
    t_hit, ok = model.evaluate(angle)
    assert math.isfinite(t_hit)
    if feasible:
        assert ok
        assert t_hit < float("inf")
