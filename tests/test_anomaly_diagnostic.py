"""
Unit tests for the anomaly-injection credit-assignment diagnostic
(GPAE paper Sec. 3.1: force one agent to take a fixed suboptimal action with
some probability, then measure dA = mean_{j != i} A_j - A_i at those steps).

Checks:
1. R_MAPPO._anomaly_advantage_gap averages the gap over injected steps only,
   ignores events where the anomaly agent is dead, and excludes dead teammates
   from the mean.
2. SMACRunner._inject_anomaly triggers only where the anomaly action is
   available, records the trigger mask, overrides the action, and replaces the
   stored log-prob with log pi(anomaly_action).

Run: python tests/test_anomaly_diagnostic.py
"""
import os
import sys
import types

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO
from onpolicy.runner.shared.smac_runner import SMACRunner


def test_anomaly_advantage_gap():
    host = types.SimpleNamespace(anomaly_agent_id=0)
    gap_fn = R_MAPPO._anomaly_advantage_gap

    T, B, N = 4, 3, 3
    adv = np.zeros((T, B, N, 1), dtype=np.float32)
    active = np.ones((T, B, N, 1), dtype=np.float32)
    anom = np.zeros((T, B, 1), dtype=np.float32)

    # event 1: agent0 = -1.0, teammates +0.5 -> gap 1.5
    anom[1, 0, 0] = 1.0
    adv[1, 0, 0, 0] = -1.0
    adv[1, 0, 1, 0] = 0.5
    adv[1, 0, 2, 0] = 0.5

    # event 2: one teammate dead -> mean over the single active one -> gap 2.0
    anom[2, 1, 0] = 1.0
    adv[2, 1, 1, 0] = 2.0
    adv[2, 1, 2, 0] = 99.0
    active[2, 1, 2, 0] = 0.0

    # event 3: anomaly agent itself dead -> excluded
    anom[3, 2, 0] = 1.0
    active[3, 2, 0, 0] = 0.0
    adv[3, 2, 1, 0] = 123.0

    gap, count = gap_fn(host, adv, anom, active)
    assert count == 2, count
    assert abs(gap - (1.5 + 2.0) / 2) < 1e-6, gap

    gap0, count0 = gap_fn(host, adv, np.zeros_like(anom), active)
    assert (gap0, count0) == (0.0, 0)
    print("test_anomaly_advantage_gap OK")


def test_inject_anomaly():
    n_envs, n_agents, act_dim = 6, 3, 5

    runner = SMACRunner.__new__(SMACRunner)  # skip full __init__ (needs SC2)
    runner.anomaly_agent_id = 1
    runner.anomaly_prob = 0.5
    runner.anomaly_action = 1
    runner.n_rollout_threads = n_envs

    buf = types.SimpleNamespace()
    buf.obs = np.zeros((2, n_envs, n_agents, 7), dtype=np.float32)
    buf.rnn_states = np.zeros((2, n_envs, n_agents, 1, 8), dtype=np.float32)
    buf.masks = np.ones((2, n_envs, n_agents, 1), dtype=np.float32)
    buf.available_actions = np.ones((2, n_envs, n_agents, act_dim), dtype=np.float32)
    buf.available_actions[0, 0, 1, 1] = 0.0  # env0: anomaly action unavailable
    buf.anomaly_masks = np.zeros((1, n_envs, 1), dtype=np.float32)
    runner.buffer = buf

    probs = np.full((n_envs, act_dim), 0.2, dtype=np.float32)
    policy = types.SimpleNamespace(get_action_probs=lambda *a, **k: torch.tensor(probs))
    runner.trainer = types.SimpleNamespace(policy=policy)

    np.random.seed(1)
    actions = np.full((n_envs, n_agents, 1), 3, dtype=np.float32)
    logp = np.full((n_envs, n_agents, 1), -0.7, dtype=np.float32)
    new_actions, new_logp = runner._inject_anomaly(0, actions, logp)

    trig = buf.anomaly_masks[0, :, 0].astype(bool)
    assert not trig[0], "env0 must not trigger (anomaly action unavailable)"
    assert trig.any(), "expected at least one trigger with p=0.5"
    assert (new_actions[trig, 1, 0] == 1).all()
    assert np.allclose(new_logp[trig, 1, 0], np.log(0.2 + 1e-10))
    assert (new_actions[~trig, 1, 0] == 3).all()
    assert (new_actions[:, 0, 0] == 3).all() and (new_actions[:, 2, 0] == 3).all()
    assert (actions[:, 1, 0] == 3).all(), "input array must not be mutated"
    print("test_inject_anomaly OK")


if __name__ == "__main__":
    test_anomaly_advantage_gap()
    test_inject_anomaly()
    print("ALL TESTS PASSED")
