"""Tests for graph observation encoding."""

from __future__ import annotations

import numpy as np
from kaggle_environments import make

from rl.graph_constants import EDGE_FEAT_DIM, GLOBAL_FEAT_DIM, MAX_NODES
from rl.graph_encoding import GraphFeatureConfig, GraphFeatureState, encode_graph_observation


def test_encode_graph_observation_shapes():
    env = make("orbit_wars", configuration={"seed": 42, "episodeSteps": 30}, debug=True)

    def noop(obs):
        return []

    env.run([noop, noop])
    obs = env.steps[1][0].observation
    cfg = GraphFeatureConfig(history_steps=3, future_steps=3)
    state = GraphFeatureState(config=cfg)
    graph, state2 = encode_graph_observation(obs, state, episode_steps=30)

    assert state2 is state
    assert graph.nodes.shape == (MAX_NODES, cfg.node_feat_dim)
    assert graph.edges.shape == (MAX_NODES, MAX_NODES, EDGE_FEAT_DIM)
    assert graph.global_features.shape == (GLOBAL_FEAT_DIM,)
    assert graph.node_valid.shape == (MAX_NODES,)
    assert graph.source_mask.shape == (MAX_NODES,)
    assert graph.slot_planet_ids.shape == (MAX_NODES,)
    assert not graph.edge_valid.any()


def test_history_advances():
    env = make("orbit_wars", configuration={"seed": 7, "episodeSteps": 20}, debug=True)

    def noop(obs):
        return []

    env.run([noop, noop])
    cfg = GraphFeatureConfig(history_steps=2, future_steps=2)
    state = GraphFeatureState(config=cfg)
    state.reset(env.steps[1][0].observation, episode_steps=20)
    g1, _ = encode_graph_observation(env.steps[1][0].observation, state, episode_steps=20)
    g2, _ = encode_graph_observation(env.steps[2][0].observation, state, episode_steps=20)
    assert not np.allclose(g1.nodes, g2.nodes)
