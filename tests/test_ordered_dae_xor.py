"""
Synthetic XOR validation for the ordered DAE head (O-DAE).

Setup: 2 agents, binary actions, one constant state, one-step episodes,
exactly uniform rollout policy (actor output layer zeroed). Team reward
r(a1, a2) = +1 if a1 == a2 else -1, so the true joint advantage equals the
reward and V* = 0.

Checks (research plan section 3.5 / 4.8):
1. additive head (F-DAE): the best additive approximation at the uniform
   policy is A_i(a_i) = E[A | a_i] = 0 -> factors collapse to ~0 and the DAE
   residual keeps ~full variance (loss plateau ~0.5 with e^2/2).
2. ordered head, fixed order (O-DAE): factors converge to the MAAD terms
   A^1(a1) = E[A | a1] = 0 and A^2(a1, a2) = A(a1, a2); loss -> ~0.
3. ordered head, random permutations (O-DAE-perm): the factor sum matches A,
   and the permutation-averaged per-agent factor -> A / 2 (Shapley value of
   the symmetric XOR game).

Run: python tests/test_ordered_dae_xor.py
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
    # exactly uniform rollout policy for clean centering / theory values
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
    # one-step episodes: the recursion bootstraps through masks[t + 1] = 0
    buffer.masks[1:] = 0.0


def team_reward(buffer):
    return buffer.rewards[:, :, 0, 0]


def train(trainer, buffer, iters=400, tag=""):
    loss = None
    for i in range(iters):
        loss, _ = trainer.dae_update(buffer)
        if (i + 1) % 100 == 0:
            print(f"  [{tag}] iter {i + 1}: dae_loss = {loss.item():.4f}")
    return loss.item()


def check(name, condition, detail):
    status = "PASS" if condition else "FAIL"
    print(f"  {status}: {name} ({detail})")
    return condition


def main():
    results = []

    print("case 1: additive head (F-DAE-agent) -- factors must collapse at the XOR symmetric point")
    trainer, buffer = build(0, dae_head="additive", dae_actor_adv="agent")
    loss = train(trainer, buffer, tag="additive")
    factors = trainer.compute_dae_advantages(buffer)
    max_factor = np.abs(factors).max()
    results.append(check("loss plateaus at residual variance",
                         abs(loss - 0.5) < 0.05, f"loss={loss:.4f}, expected ~0.5"))
    results.append(check("factors ~ 0", max_factor < 0.15, f"max |A_i|={max_factor:.4f}"))

    print("case 2: ordered head, fixed order (O-DAE) -- factors must match the MAAD terms")
    trainer, buffer = build(1, dae_head="ordered", dae_actor_adv="agent", dae_order="fixed")
    loss = train(trainer, buffer, tag="ordered-fixed")
    factors = trainer.compute_dae_advantages(buffer)
    reward = team_reward(buffer)
    err_first = np.abs(factors[:, :, 0, 0]).max()
    err_second = np.abs(factors[:, :, 1, 0] - reward).max()
    results.append(check("loss -> 0", loss < 0.02, f"loss={loss:.4f}"))
    results.append(check("first agent factor ~ 0 (E[A|a1] = 0)",
                         err_first < 0.15, f"max |A^1|={err_first:.4f}"))
    results.append(check("second agent factor ~ A(a1, a2)",
                         err_second < 0.2, f"max |A^2 - A|={err_second:.4f}"))

    print("case 3: ordered head, random permutations (O-DAE-perm) -- Shapley averaging")
    trainer, buffer = build(2, dae_head="ordered", dae_actor_adv="agent",
                            dae_order="permute", dae_perm_eval_samples=32)
    loss = train(trainer, buffer, iters=600, tag="ordered-perm")
    reward = team_reward(buffer)
    factors = trainer.compute_dae_advantages(buffer)  # averaged over 32 permutations
    shapley_err = np.abs(factors[:, :, :, 0] - reward[:, :, None] / 2.0).mean()
    # a single permutation must still sum to the joint advantage
    trainer.dae_perm_eval_samples = 1
    single = trainer.compute_dae_advantages(buffer)
    sum_err = np.abs(single.sum(axis=2)[:, :, 0] - reward).max()
    results.append(check("loss -> 0", loss < 0.02, f"loss={loss:.4f}"))
    results.append(check("factor sum ~ A under one permutation",
                         sum_err < 0.25, f"max |sum - A|={sum_err:.4f}"))
    results.append(check("permutation-averaged factors ~ A / 2 (Shapley)",
                         shapley_err < 0.15, f"mean |phi_i - A/2|={shapley_err:.4f}"))

    print()
    if all(results):
        print(f"ALL {len(results)} CHECKS PASSED")
        return 0
    print(f"{results.count(False)}/{len(results)} CHECKS FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
