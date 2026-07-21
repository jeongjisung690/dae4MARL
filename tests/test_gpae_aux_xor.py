"""
Synthetic XOR validation for the GPAE auxiliary loss on the ordered DAE head.

Setup (same as test_ordered_dae_xor): 2 agents, binary actions, one constant
state, one-step episodes, exactly uniform rollout policy. Team reward
r(a1, a2) = +1 if a1 == a2 else -1, so V* = 0 and Q(a) = r(a).

GPAE theory values at the uniform policy:
- E_Q^k(a_-k) = E_{a_k}[r] = 0 for every a_-k  -> the eq head must go to ~0.
- GPAE advantage = r - E_Q^k = r (one-step episodes, lambda irrelevant).
- The fully conditioned centered factor A_k(s, a_-k, a_k) must converge to
  r(a) for BOTH agents. Under dae_order=fixed the main residual loss never
  trains agent 0 with a non-empty prefix, so agent 0's full-prefix factor is
  supervised ONLY by the auxiliary loss - this isolates the new gradient path.
- The main residual loss must still reach ~0 (aux must not fight it).

Run: python tests/test_gpae_aux_xor.py
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


def build(rng_seed, **overrides):
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
    fill_buffer(buffer, np.random.default_rng(rng_seed))
    return trainer, buffer


def fill_buffer(buffer, rng):
    actions = rng.integers(0, 2, size=(EPISODE_LENGTH, N_THREADS, NUM_AGENTS, 1))
    xor_reward = np.where(actions[:, :, 0, 0] == actions[:, :, 1, 0], 1.0, -1.0).astype(np.float32)
    buffer.share_obs[:] = 1.0
    buffer.obs[:] = 1.0
    buffer.actions[:] = actions
    buffer.rewards[:] = xor_reward[:, :, None, None]
    buffer.masks[1:] = 0.0


def full_prefix_eval(trainer, buffer):
    """E_Q^k and the full-prefix centered factor per (t, b, k), raw units."""
    share_obs, rnn_states_critic, masks = trainer._dae_flat_inputs(buffer)
    own_index, own_actions = trainer._factor_row_inputs(buffer)
    own_probs = trainer._policy_probs_buffer(buffer).reshape(-1, trainer.policy.critic.action_dim)
    prefix_onehot, prefix_mask = trainer._full_prefix_inputs(buffer, None)
    with torch.no_grad():
        eq, factors = trainer.policy.evaluate_gpae_sequence(
            share_obs, rnn_states_critic, masks, own_index, own_actions,
            own_probs, prefix_onehot, prefix_mask, trainer.data_chunk_length)
    shape = (EPISODE_LENGTH, N_THREADS, NUM_AGENTS)
    return eq.numpy().reshape(shape), factors.numpy().reshape(shape)


def check(name, condition, detail):
    status = "PASS" if condition else "FAIL"
    print(f"  {status}: {name} ({detail})")
    return condition


def main():
    results = []

    print("GPAE aux on O-DAE-fixed: last-position factors must be grounded to r for both agents")
    trainer, buffer = build(0, dae_head="ordered", dae_actor_adv="agent",
                            dae_order="fixed", dae_gpae_coef=1.0)
    loss = None
    for i in range(600):
        loss, _ = trainer.dae_update(buffer)
        if (i + 1) % 150 == 0:
            print(f"  iter {i + 1}: dae_loss = {loss.item():.4f}, "
                  f"eq_loss = {trainer._last_gpae_eq_loss:.4f}, "
                  f"aux_loss = {trainer._last_gpae_aux_loss:.4f}, "
                  f"corr = {trainer._last_gpae_factor_corr:.3f}")

    reward = buffer.rewards[:, :, 0, 0]
    eq, factors = full_prefix_eval(trainer, buffer)

    results.append(check("main residual loss -> 0", loss.item() < 0.02,
                         f"loss={loss.item():.4f}"))
    results.append(check("E_Q head -> 0 (E_a_k[r] = 0)", np.abs(eq).max() < 0.15,
                         f"max |E_Q|={np.abs(eq).max():.4f}"))
    err0 = np.abs(factors[:, :, 0] - reward).mean()
    err1 = np.abs(factors[:, :, 1] - reward).mean()
    results.append(check("agent 0 full-prefix factor ~ r (aux-only gradient path)",
                         err0 < 0.1, f"mean |A^0 - r|={err0:.4f}"))
    results.append(check("agent 1 full-prefix factor ~ r",
                         err1 < 0.1, f"mean |A^1 - r|={err1:.4f}"))
    results.append(check("factor/target correlation ~ 1",
                         trainer._last_gpae_factor_corr > 0.95,
                         f"corr={trainer._last_gpae_factor_corr:.3f}"))

    # the sampled-order factors consumed by the actor must still satisfy the
    # MAAD fixed-order solution (A^1(a1) = 0, A^2(a1, a2) = r): the aux loss
    # must not have distorted the main decomposition.
    sampled = trainer.compute_dae_advantages(buffer)
    err_first = np.abs(sampled[:, :, 0, 0]).max()
    err_second = np.abs(sampled[:, :, 1, 0] - reward).max()
    results.append(check("fixed-order factor of agent 0 still ~ 0",
                         err_first < 0.15, f"max |A^1|={err_first:.4f}"))
    results.append(check("fixed-order factor of agent 1 still ~ r",
                         err_second < 0.25, f"max |A^2 - r|={err_second:.4f}"))

    print()
    if all(results):
        print(f"ALL {len(results)} CHECKS PASSED")
        return 0
    print(f"{results.count(False)}/{len(results)} CHECKS FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
