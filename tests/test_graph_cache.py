"""Tests for graph edge and fleet caches."""

from __future__ import annotations

from kaggle_environments import make

from rl.graph_cache import EdgeTravelCache, FleetProjectionCache
from rl.graph_constants import SHIP_BUCKETS
from rl.graph_encoding import build_slot_mapping


def test_edge_cache_ship_buckets():
    env = make("orbit_wars", configuration={"seed": 3, "episodeSteps": 40}, debug=True)

    def noop(obs):
        return []

    env.run([noop, noop])
    obs = env.steps[1][0].observation
    cache = EdgeTravelCache(episode_steps=40, coarse=8)
    sm = cache.reset(obs)
    feats = cache.edge_features_for_pair(1, 0, 1, obs)
    assert feats.shape == (len(SHIP_BUCKETS) * 4,)
    assert sm.node_valid.sum() >= 1


def test_fleet_projection_cache_empty():
    env = make("orbit_wars", configuration={"seed": 5, "episodeSteps": 20}, debug=True)

    def noop(obs):
        return []

    env.run([noop, noop])
    obs = env.steps[1][0].observation
    sm = build_slot_mapping(obs)
    fc = FleetProjectionCache()
    fc.reset(sm, obs)
    fc.update(obs, sm)
    friendly, enemy = fc.incoming_by_slot(int(obs.step), int(obs.player), horizon=3)
    assert friendly.shape == (3, len(sm.slot_planet_ids))
