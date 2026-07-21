"""
Synthetic XOR validation for off-policy replay of the DAE / GPAE head training.

Setup: 2 agents, binary actions, one constant state, one-step episodes, uniform
current policy pi (actor output layer zeroed). Team reward r = +1 if a1 == a2
else -1, so V*_pi = 0, Q(a) = r(a), E_Q^k(a_-k) = E_{a_k ~ pi}[r] = 0.

Checks:
1. On-policy equivalence: a replay head update with mu = pi (all importance
   ratios 1) must produce EXACTLY the same critic parameter update as the
   regular fresh update -- the V-trace residual recursion and the DT-ISR GPAE
   traces reduce to the on-policy forms at rho = 1.
2. Off-policy correctness with a biased behavior policy mu(a=0) = 0.8:
   - E_mu[r] = 0.36, but the difference-reward recursion is action-distribution
     free, so V must stay ~0 (no bias leak from the replayed data).
   - The sampled-order factors must still solve the MAAD decomposition
     (A^1 ~ 0, A^2 ~ r).
   - The E_Q head must converge to the analytic truncated-IS fixed point
     E_Q(a_-k) = sum_a min(mu, pi) r / sum_a min(mu, pi) = +-3/7
     (the known V-trace-style bias toward mu; validates the rho_bar wiring).

Run: python tests/test_dae_replay_xor.py
"""
import os
import sys

import numpy as np
import torch
from gym import spaces

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from onpolicy.config import get_config
from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from onpolicy.utils.shared_buffer import SharedReplayBuffer

NUM_AGENTS = 2
EPISODE_LENGTH = 32
N_THREADS = 256
OBS_DIM = 3


def make_args(**overrides):
    parser = get_config()
    args = parser.parse_args([])
    args.num_agents = NUM_AGENTS
    args.episode_length = EPISODE_LENGTH
    args.n_rollout_threads = N_THREADS
    args.hidden_size = 64
    args.layer_N = 1
    args.use_recurrent_policy = False
    args.use_naive_recurrent_policy = False
    args.use_popart = False
    args.use_valuenorm = False
    args.use_dae = True
    args.dae_head_hidden_size = 64
    args.dae_normalize_advantages = False
    args.dae_centering = "exact"
    args.dae_num_mini_batch = 1
    args.critic_lr = 5e-3
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def build(rng_seed, mu0=0.5, **overrides):
    torch.manual_seed(rng_seed)
    np.random.seed(rng_seed)
    args = make_args(**overrides)
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
    act_space = spaces.Discrete(2)
    policy = R_MAPPOPolicy(args, obs_space, obs_space, act_space, device=torch.device("cpu"))
    for p in policy.actor.act.parameters():
        p.data.zero_()
    trainer = R_MAPPO(args, policy, device=torch.device("cpu"))
    buffer = SharedReplayBuffer(args, NUM_AGENTS, obs_space, obs_space, act_space)
    fill_buffer(buffer, np.random.default_rng(rng_seed), mu0)
    return trainer, buffer


def fill_buffer(buffer, rng, mu0):
    """Actions sampled per agent from the behavior policy mu(a=0) = mu0."""
    actions = (rng.random((EPISODE_LENGTH, N_THREADS, NUM_AGENTS, 1)) > mu0).astype(np.int64)
    xor_reward = np.where(actions[:, :, 0, 0] == actions[:, :, 1, 0], 1.0, -1.0).astype(np.float32)
    buffer.share_obs[:] = 1.0
    buffer.obs[:] = 1.0
    buffer.actions[:] = actions
    mu = np.where(actions == 0, mu0, 1.0 - mu0).astype(np.float32)
    buffer.action_log_probs[:] = np.log(mu)
    buffer.rewards[:] = xor_reward[:, :, None, None]
    buffer.masks[1:] = 0.0


def full_prefix_eval(trainer, snap):
    probs = trainer._policy_probs_replay(snap)
    own_index, own_actions = trainer._factor_row_inputs(snap)
    share_obs, rnn_states_critic, masks = trainer._dae_flat_inputs(snap)
    prefix_onehot, prefix_mask = trainer._full_prefix_inputs(snap, None)
    with torch.no_grad():
        eq, factors = trainer.policy.evaluate_gpae_sequence(
            share_obs, rnn_states_critic, masks, own_index, own_actions,
            probs.reshape(-1, 2), prefix_onehot, prefix_mask, trainer.data_chunk_length)
    shape = (EPISODE_LENGTH, N_THREADS, NUM_AGENTS)
    return eq.numpy().reshape(shape), factors.numpy().reshape(shape)


def check(name, condition, detail):
    status = "PASS" if condition else "FAIL"
    print(f"  {status}: {name} ({detail})")
    return condition


def main():
    results = []
    common = dict(dae_head="ordered", dae_actor_adv="agent", dae_order="fixed",
                  dae_gpae_coef=1.0, dae_replay_size=4, dae_replay_updates=1)

    print("case 1: replay update with mu = pi must equal the fresh on-policy update")
    trainer_a, buffer = build(0, mu0=0.5, **common)
    trainer_b, _ = build(0, mu0=0.5, **common)
    for pa, pb in zip(trainer_a.policy.critic.parameters(), trainer_b.policy.critic.parameters()):
        assert torch.equal(pa, pb), "trainers must start from identical weights"

    trainer_a._dae_update_minibatch(buffer)
    snap = trainer_b._make_dae_snapshot(buffer)
    trainer_b._dae_replay_minibatch(snap)

    max_diff = max((pa - pb).abs().max().item()
                   for pa, pb in zip(trainer_a.policy.critic.parameters(),
                                     trainer_b.policy.critic.parameters()))
    results.append(check("critic params identical after one update",
                         max_diff < 1e-6, f"max param diff={max_diff:.2e}"))
    results.append(check("importance ratios were 1",
                         abs(trainer_b._last_replay_rho_own_mean - 1.0) < 1e-5,
                         f"mean rho_own={trainer_b._last_replay_rho_own_mean:.6f}"))

    print("case 2: replay-only training from a biased behavior policy mu(a=0) = 0.8")
    trainer, buffer = build(1, mu0=0.8, **common)
    snap = trainer._make_dae_snapshot(buffer)
    reward = buffer.rewards[:, :, 0, 0]
    print(f"  (E_mu[r] = {reward.mean():.3f} -- a naive return target would push V there)")
    loss = None
    for i in range(800):
        loss = trainer._dae_replay_minibatch(snap)
        if (i + 1) % 200 == 0:
            print(f"  iter {i + 1}: replay_loss = {loss.item():.4f}, "
                  f"c_joint = {trainer._last_replay_c_joint_mean:.3f}")

    with torch.no_grad():
        probs = trainer._policy_probs_replay(snap)
        values, factors = trainer._evaluate_dae_buffer(snap, policy_probs=probs)
    values = values.numpy()
    factors = factors.numpy()

    results.append(check("V stays ~0 despite E_mu[r] > 0 (no off-policy bias leak)",
                         np.abs(values).max() < 0.1,
                         f"max |V|={np.abs(values).max():.4f} vs E_mu[r]={reward.mean():.3f}"))
    err_first = np.abs(factors[:, :, 0, 0]).max()
    err_second = np.abs(factors[:, :, 1, 0] - reward).max()
    results.append(check("fixed-order factor of agent 0 ~ 0",
                         err_first < 0.15, f"max |A^1|={err_first:.4f}"))
    results.append(check("fixed-order factor of agent 1 ~ r",
                         err_second < 0.25, f"max |A^2 - r|={err_second:.4f}"))

    # analytic truncated-IS fixed point of the E_Q head:
    # E_Q(a_-k) = (min(.8,.5) * r(0) + min(.2,.5) * r(1)) / .7 = +-3/7
    eq, _ = full_prefix_eval(trainer, snap)
    other_action = snap.actions[:, :, ::-1, 0]  # a_-k for k = 0, 1
    eq_expected = np.where(other_action == 0, 3.0 / 7.0, -3.0 / 7.0)
    eq_err = np.abs(eq - eq_expected).mean()
    results.append(check("E_Q converges to the analytic truncated-IS fixed point (+-3/7)",
                         eq_err < 0.1, f"mean |E_Q - fp|={eq_err:.4f}"))

    print("case 3: recurrent actor -- recomputed sequence probs must match rollout probs (rho = 1)")
    torch.manual_seed(3)
    np.random.seed(3)
    args = make_args(dae_head="ordered", dae_actor_adv="agent", dae_order="fixed",
                     dae_gpae_coef=1.0, dae_replay_size=2, dae_replay_updates=1,
                     use_recurrent_policy=True, data_chunk_length=8)
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
    act_space = spaces.Discrete(2)
    # random (NOT zeroed) actor: a nontrivial, state-history-dependent distribution
    policy = R_MAPPOPolicy(args, obs_space, obs_space, act_space, device=torch.device("cpu"))
    trainer = R_MAPPO(args, policy, device=torch.device("cpu"))
    buffer = SharedReplayBuffer(args, NUM_AGENTS, obs_space, obs_space, act_space)
    rng = np.random.default_rng(3)
    # varying observations + a mid-sequence episode reset exercise the RNN rebuild
    buffer.share_obs[:] = rng.standard_normal(buffer.share_obs.shape).astype(np.float32)
    buffer.obs[:] = buffer.share_obs.copy()
    buffer.masks[16] = 0.0

    with torch.no_grad():
        for step in range(EPISODE_LENGTH):
            _, actions, log_probs, rnn_states, rnn_states_critic = policy.get_actions(
                np.concatenate(buffer.share_obs[step]),
                np.concatenate(buffer.obs[step]),
                np.concatenate(buffer.rnn_states[step]),
                np.concatenate(buffer.rnn_states_critic[step]),
                np.concatenate(buffer.masks[step]))
            buffer.actions[step] = np.array(np.split(actions.numpy(), N_THREADS))
            buffer.action_log_probs[step] = np.array(np.split(log_probs.numpy(), N_THREADS))
            buffer.rnn_states[step + 1] = np.array(np.split(rnn_states.numpy(), N_THREADS))
            buffer.rnn_states_critic[step + 1] = np.array(np.split(rnn_states_critic.numpy(), N_THREADS))

    snap = trainer._make_dae_snapshot(buffer)
    probs = trainer._policy_probs_replay(snap)
    rho_own, rho_others, rho_joint = trainer._replay_ratios(snap, None, probs)
    rho_err = (rho_own - 1.0).abs().max().item()
    results.append(check("rho_own = 1 under an unchanged recurrent actor",
                         rho_err < 1e-4, f"max |rho - 1|={rho_err:.2e}"))

    print()
    if all(results):
        print(f"ALL {len(results)} CHECKS PASSED")
        return 0
    print(f"{results.count(False)}/{len(results)} CHECKS FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
