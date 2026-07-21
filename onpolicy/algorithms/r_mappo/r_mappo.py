import types
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from onpolicy.utils.util import get_gard_norm, huber_loss, mse_loss
from onpolicy.utils.valuenorm import ValueNorm
from onpolicy.algorithms.utils.util import check

class R_MAPPO():
    """
    Trainer class for MAPPO to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self,
                 args,
                 policy,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta
        self.gamma = args.gamma

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_dae = getattr(args, "use_dae", False)
        self.dae_epoch = getattr(args, "dae_epoch", 1)
        self.dae_num_mini_batch = getattr(args, "dae_num_mini_batch", 1)
        self.dae_loss_coef = getattr(args, "dae_loss_coef", 1.0)
        self.dae_normalize_advantages = getattr(args, "dae_normalize_advantages", False)
        self.dae_centering = getattr(args, "dae_centering", "exact")
        self.dae_warmup_updates = getattr(args, "dae_warmup_updates", 0)
        self.dae_head = getattr(args, "dae_head", "additive")
        self.dae_actor_adv = getattr(args, "dae_actor_adv", "joint")
        self.dae_order = getattr(args, "dae_order", "fixed")
        self.dae_adv_norm_scope = getattr(args, "dae_adv_norm_scope", "global")
        self.dae_perm_eval_samples = getattr(args, "dae_perm_eval_samples", 1)
        self.anomaly_agent_id = getattr(args, "anomaly_agent_id", -1)
        self.dae_gpae_coef = getattr(args, "dae_gpae_coef", 0.0)
        self.dae_gpae_lambda = getattr(args, "dae_gpae_lambda", 0.95)
        self._use_gpae_aux = self._use_dae and self.dae_gpae_coef > 0.0
        self.dae_replay_size = getattr(args, "dae_replay_size", 0)
        self.dae_replay_updates = getattr(args, "dae_replay_updates", 0)
        self.dae_trace_eta = getattr(args, "dae_trace_eta", 1.05)
        self._use_dae_replay = (self._use_dae and self.dae_replay_size > 0
                                and self.dae_replay_updates > 0)
        if self._use_dae_replay:
            self._dae_replay = deque(maxlen=self.dae_replay_size)
        if self._use_gpae_aux:
            if self.dae_head != "ordered" or self.dae_centering != "exact":
                raise ValueError("dae_gpae_coef > 0 requires dae_head=ordered and dae_centering=exact: "
                                 "the GPAE target is a centered fully-conditioned advantage, so only the "
                                 "ordered head's centered last-position factor can be tied to it.")
        # factor mode: the head is evaluated per agent row (own factor) instead
        # of once as a joint sum; needed whenever factors are consumed
        # individually (agent-wise actor signal) or conditioned on a prefix.
        self._dae_factor_mode = self.dae_head == "ordered" or self.dae_actor_adv == "agent"
        if self._use_dae and self.dae_actor_adv == "agent" and self.dae_centering != "exact":
            raise ValueError("dae_actor_adv=agent requires dae_centering=exact: without centering, "
                             "per-agent factors are identified only up to state-dependent shifts.")
        self._dae_train_calls = 0
        
        assert (self._use_popart and self._use_valuenorm) == False, ("self._use_popart and self._use_valuenorm can not be set True simultaneously")
        
        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

    def _env_slice(self, array, env_indices=None, include_last=True):
        time_slice = slice(None) if include_last else slice(None, -1)
        if env_indices is None:
            return array[time_slice]
        return array[time_slice, env_indices]

    def _joint_actions(self, buffer, env_indices=None):
        actions = self._env_slice(buffer.actions, env_indices, include_last=True)
        joint_actions = np.repeat(actions[:, :, None, :, :], buffer.num_agents, axis=2)
        return joint_actions.reshape(-1, buffer.num_agents, buffer.actions.shape[-1])

    def _dae_flat_inputs(self, buffer, env_indices=None):
        share_obs = self._env_slice(buffer.share_obs, env_indices, include_last=False).reshape(
            -1, *buffer.share_obs.shape[3:])
        # Only the hidden states at the start of the sequence are used; the critic
        # recomputes the rest in the forward pass (truncated BPTT), so they never go stale.
        rnn_states_critic = buffer.rnn_states_critic[0]
        if env_indices is not None:
            rnn_states_critic = rnn_states_critic[env_indices]
        rnn_states_critic = rnn_states_critic.reshape(-1, *buffer.rnn_states_critic.shape[3:])
        masks = self._env_slice(buffer.masks, env_indices, include_last=False).reshape(-1, 1)
        return share_obs, rnn_states_critic, masks

    def _sample_agent_orders(self, n_envs, num_agents):
        if self.dae_order == "permute":
            return np.stack([np.random.permutation(num_agents) for _ in range(n_envs)])
        return np.tile(np.arange(num_agents), (n_envs, 1))

    def _prefix_inputs_from_precede(self, buffer, env_indices, precede):
        """
        Build the ordered head's conditioning inputs, aligned with the buffer's
        (T, envs, agents) row layout: row (t, b, k) receives the one-hot actions
        of the agents j with precede[b, k, j] = 1 plus a mask marking them.
        """
        actions = self._env_slice(buffer.actions, env_indices, include_last=True)
        episode_length, n_envs, num_agents = actions.shape[:3]
        action_dim = self.policy.critic.action_dim
        act_idx = actions[..., 0].astype(np.int64).clip(0, action_dim - 1)
        onehot = np.zeros((episode_length, n_envs, num_agents, action_dim), dtype=np.float32)
        np.put_along_axis(onehot, act_idx[..., None], 1.0, axis=-1)
        prefix_onehot = onehot[:, :, None, :, :] * precede[None, :, :, :, None]
        prefix_onehot = prefix_onehot.reshape(episode_length * n_envs * num_agents,
                                              num_agents * action_dim)
        prefix_mask = np.broadcast_to(precede[None],
                                      (episode_length, n_envs, num_agents, num_agents))
        prefix_mask = np.ascontiguousarray(prefix_mask).reshape(-1, num_agents)
        return prefix_onehot, prefix_mask

    def _ordered_prefix_inputs(self, buffer, env_indices, orders):
        # ranks[b, k]: position of agent k in orders[b];
        # precede[b, i, j] = 1 iff agent j acts before agent i.
        ranks = np.argsort(orders, axis=1)
        precede = (ranks[:, None, :] < ranks[:, :, None]).astype(np.float32)
        return self._prefix_inputs_from_precede(buffer, env_indices, precede)

    def _full_prefix_inputs(self, buffer, env_indices):
        """Last-position conditioning: every other agent counts as a predecessor."""
        n_envs = buffer.n_rollout_threads if env_indices is None else len(env_indices)
        num_agents = buffer.num_agents
        precede = np.broadcast_to(1.0 - np.eye(num_agents, dtype=np.float32),
                                  (n_envs, num_agents, num_agents))
        return self._prefix_inputs_from_precede(buffer, env_indices, precede)

    def _factor_row_inputs(self, buffer, env_indices=None):
        actions = self._env_slice(buffer.actions, env_indices, include_last=True)
        episode_length, n_envs, num_agents = actions.shape[:3]
        own_actions = actions.reshape(-1, actions.shape[-1])[:, :1]
        own_index = np.tile(np.arange(num_agents), episode_length * n_envs)
        return own_index, own_actions

    def _value_norm_std(self):
        if self.value_normalizer is None:
            return None
        if hasattr(self.value_normalizer, "running_mean_var"):
            _, var = self.value_normalizer.running_mean_var()
        else:
            _, var = self.value_normalizer.debiased_mean_var()
        return torch.sqrt(var).reshape(1)

    def _evaluate_dae_buffer(self, buffer, env_indices=None, policy_probs=None):
        """
        Evaluate the DAE head over the buffer. Returns (values, advantages),
        both (T, n_envs, num_agents, 1). In factor mode advantages[t, b, k] is
        agent k's own centered factor; otherwise it is the summed joint
        advantage, identical across the agent dimension.
        :param policy_probs: optional (T, n_envs, num_agents, action_dim) current-policy
                             probabilities to use for exact centering instead of
                             recomputing them from the buffer's stored per-step
                             actor hidden states (required for replayed data,
                             whose stored hidden states are stale).
        """
        share_obs, rnn_states_critic, masks = self._dae_flat_inputs(buffer, env_indices)
        # Centering probs use per-step stored actor hidden states; this is exact
        # only because DAE updates run before any actor update (actor == mu).
        if self._dae_factor_mode:
            own_index, own_actions = self._factor_row_inputs(buffer, env_indices)
            if self.dae_centering == "exact":
                if policy_probs is None:
                    policy_probs = self._policy_probs_buffer(buffer, env_indices)
                own_probs = policy_probs.reshape(-1, self.policy.critic.action_dim)
            else:
                own_probs = None
            prefix_onehot = prefix_mask = None
            if self.dae_head == "ordered":
                n_envs = buffer.n_rollout_threads if env_indices is None else len(env_indices)
                orders = self._sample_agent_orders(n_envs, buffer.num_agents)
                prefix_onehot, prefix_mask = self._ordered_prefix_inputs(buffer, env_indices, orders)
            values, advantages = self.policy.evaluate_factor_dae_sequence(share_obs,
                                                                          rnn_states_critic,
                                                                          masks,
                                                                          own_index,
                                                                          own_actions,
                                                                          own_probs,
                                                                          prefix_onehot,
                                                                          prefix_mask,
                                                                          self.data_chunk_length)
        else:
            joint_actions = self._joint_actions(buffer, env_indices)
            if self.dae_centering == "exact":
                if policy_probs is None:
                    policy_probs = self._policy_probs_buffer(buffer, env_indices)
                joint_action_probs = policy_probs.reshape(
                    -1, buffer.num_agents, self.policy.critic.action_dim
                ).repeat_interleave(buffer.num_agents, dim=0)
            else:
                joint_action_probs = None
            values, advantages = self.policy.evaluate_dae_sequence(share_obs,
                                                                   rnn_states_critic,
                                                                   masks,
                                                                   joint_actions,
                                                                   joint_action_probs,
                                                                   self.data_chunk_length)
        # The advantage head outputs live in normalized units; scale by the value
        # normalizer std so raw-space advantages get the same gradient scale as the
        # value head in the normalized DAE loss (PopArt-style parity).
        adv_std = self._value_norm_std()
        if adv_std is not None:
            advantages = advantages * adv_std
        n_envs = buffer.n_rollout_threads if env_indices is None else len(env_indices)
        values = values.view(buffer.episode_length,
                             n_envs,
                             buffer.num_agents,
                             1)
        advantages = advantages.view_as(values)
        return values, advantages

    @torch.no_grad()
    def _policy_probs_buffer(self, buffer, env_indices=None):
        obs = self._env_slice(buffer.obs, env_indices, include_last=False).reshape(-1, *buffer.obs.shape[3:])
        rnn_states = self._env_slice(buffer.rnn_states, env_indices, include_last=False).reshape(
            -1, *buffer.rnn_states.shape[3:])
        masks = self._env_slice(buffer.masks, env_indices, include_last=False).reshape(-1, 1)
        if buffer.available_actions is None:
            available_actions = None
        else:
            available_actions = self._env_slice(buffer.available_actions, env_indices, include_last=False).reshape(
                -1, buffer.available_actions.shape[-1])

        probs = self.policy.get_action_probs(obs, rnn_states, masks, available_actions)
        n_envs = buffer.n_rollout_threads if env_indices is None else len(env_indices)
        return probs.view(buffer.episode_length,
                          n_envs,
                          buffer.num_agents,
                          -1).detach()

    def _masked_advantage_stats(self, advantages, active_masks):
        advantages_copy = advantages.copy()
        advantages_copy[active_masks == 0.0] = np.nan
        mean = np.nanmean(advantages_copy)
        std = np.nanstd(advantages_copy)
        rms = np.sqrt(np.nanmean(np.square(advantages_copy)))
        return float(mean), float(std), float(rms)

    def _reference_gae_advantages(self, buffer):
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]

        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        return (advantages - mean_advantages) / (std_advantages + 1e-5)

    def _advantage_alignment_stats(self, dae_advantages, gae_advantages, active_masks):
        active = active_masks != 0.0
        dae = dae_advantages[active].astype(np.float64)
        gae = gae_advantages[active].astype(np.float64)
        finite = np.isfinite(dae) & np.isfinite(gae)
        dae = dae[finite]
        gae = gae[finite]

        if dae.size < 2 or np.std(dae) < 1e-8 or np.std(gae) < 1e-8:
            return 0.0, 0.0, 0.0

        corr = np.corrcoef(dae, gae)[0, 1]
        sign_agreement = np.mean(np.sign(dae) == np.sign(gae))
        cosine = np.dot(dae, gae) / ((np.linalg.norm(dae) * np.linalg.norm(gae)) + 1e-8)
        return float(corr), float(sign_agreement), float(cosine)

    @torch.no_grad()
    def _collect_dae_advantages(self, buffer):
        if self.dae_num_mini_batch <= 1 or buffer.n_rollout_threads <= 1:
            _, advantages = self._evaluate_dae_buffer(buffer)
            return advantages.cpu().numpy()

        advantages = np.zeros((buffer.episode_length,
                               buffer.n_rollout_threads,
                               buffer.num_agents,
                               1), dtype=np.float32)
        indices = np.arange(buffer.n_rollout_threads)
        mini_batches = np.array_split(indices, min(self.dae_num_mini_batch, buffer.n_rollout_threads))
        for env_indices in mini_batches:
            if len(env_indices) == 0:
                continue
            _, batch_advantages = self._evaluate_dae_buffer(buffer, env_indices)
            advantages[:, env_indices] = batch_advantages.cpu().numpy()
        return advantages

    def _anomaly_advantage_gap(self, advantages, anomaly_masks, active_masks):
        """
        Credit-assignment diagnostic (GPAE paper Sec. 3.1): at steps where the
        anomaly agent's action was overridden with a fixed suboptimal action,
        dA = mean_{j != i, active} A_j - A_i on the actor's advantage signal.
        Positive gap = the estimator penalizes the misbehaving agent relative to
        its teammates. Returns (mean gap, number of events).
        """
        i = self.anomaly_agent_id
        adv = advantages[..., 0]
        active = active_masks[..., 0] != 0.0
        events = (anomaly_masks[..., 0] != 0.0) & active[:, :, i]

        others = np.ones(adv.shape[2], dtype=bool)
        others[i] = False
        others_active = active[:, :, others]
        n_others = others_active.sum(axis=-1)
        valid = events & (n_others > 0)
        if not valid.any():
            return 0.0, 0
        mean_others = (adv[:, :, others] * others_active).sum(axis=-1) / np.maximum(n_others, 1)
        gap = mean_others - adv[:, :, i]
        return float(gap[valid].mean()), int(valid.sum())

    def _factor_rms_spread(self, factors, active_masks):
        """Dispersion of per-agent factor scales: std over agents of each
        agent's masked RMS, divided by their mean."""
        mask = (active_masks != 0.0).astype(np.float64)
        denom = mask.sum(axis=(0, 1)).reshape(-1) + 1e-8
        agent_ms = (np.square(factors) * mask).sum(axis=(0, 1)).reshape(-1) / denom
        agent_rms = np.sqrt(agent_ms)
        mean = agent_rms.mean()
        if mean < 1e-8:
            return 0.0
        return float(agent_rms.std() / mean)

    def _agent_rms_normalize(self, advantages, active_masks):
        mask = (active_masks != 0.0).astype(np.float64)
        denom = mask.sum(axis=(0, 1), keepdims=True) + 1e-8
        agent_rms = np.sqrt((np.square(advantages) * mask).sum(axis=(0, 1), keepdims=True) / denom)
        return advantages / (agent_rms + 1e-5)

    @torch.no_grad()
    def compute_dae_advantages(self, buffer):
        n_perm_samples = 1
        if self.dae_head == "ordered" and self.dae_order == "permute":
            n_perm_samples = max(1, self.dae_perm_eval_samples)
        advantages = self._collect_dae_advantages(buffer)
        for _ in range(n_perm_samples - 1):
            advantages = advantages + self._collect_dae_advantages(buffer)
        if n_perm_samples > 1:
            advantages = advantages / n_perm_samples

        if self._dae_factor_mode:
            self.dae_factor_rms_spread = self._factor_rms_spread(advantages, buffer.active_masks[:-1])
            if self.dae_actor_adv == "joint":
                advantages = np.broadcast_to(advantages.sum(axis=2, keepdims=True),
                                             advantages.shape).copy()
        else:
            self.dae_factor_rms_spread = 0.0

        advantages[buffer.active_masks[:-1] == 0.0] = 0.0
        raw_mean, raw_std, raw_rms = self._masked_advantage_stats(advantages, buffer.active_masks[:-1])
        self.dae_adv_raw_mean = raw_mean
        self.dae_adv_raw_std = raw_std
        self.dae_adv_raw_rms = raw_rms

        if self.dae_normalize_advantages:
            if (self.dae_adv_norm_scope == "agent" and self._dae_factor_mode
                    and self.dae_actor_adv == "agent"):
                advantages = self._agent_rms_normalize(advantages, buffer.active_masks[:-1])
            else:
                advantages = advantages / (raw_rms + 1e-5)

        return advantages

    def _gpae_losses(self, buffer, env_indices, rewards, masks, bad_masks, active_masks, bootstrap,
                     policy_probs=None, rho_own=None, rho_others=None):
        """
        GPAE auxiliary losses (Kim et al., AAMAS 2026).

        A dedicated head estimates the counterfactual value
        E_Q^k(s, a_-k) = E_{a_k ~ pi_k}[Q(s, a_k, a_-k)]. Its per-agent TD errors
        delta^k_t = r_t + gamma * E_Q^k_{t+1} - E_Q^k_t accumulate into the
        n-step GPAE advantage. Two losses come out:
        - eq_loss: value-iteration loss of the E_Q head itself
          (target E_Q + rho_bar * A_gpae; on fresh data rho = 1 exactly since
          DAE updates run before any actor update);
        - aux_loss: ties the ordered head's fully conditioned (last-position)
          centered factor A_k(s, a_-k, a_k) to the reward-grounded GPAE
          advantage. This is the only place the per-agent SPLIT of the DAE
          decomposition receives direct supervision.

        On-policy (rho_own None): trace c = lambda, the paper's on-policy case.
        Off-policy (replay): DT-ISR traces c^k = lambda * min(1, rho^k * min(eta, rho^-k))
        weight the TD terms, rho_bar^k = min(1, rho^k) scales the eq target
        increment (paper Eq. 11) and down-weights the aux supervision where the
        executed own action is improbable under the current policy.
        Returns (eq_loss, aux_loss); diagnostics stored on self.
        """
        share_obs, rnn_states_critic, seq_masks = self._dae_flat_inputs(buffer, env_indices)
        own_index, own_actions = self._factor_row_inputs(buffer, env_indices)
        if policy_probs is None:
            policy_probs = self._policy_probs_buffer(buffer, env_indices)
        own_probs = policy_probs.reshape(-1, self.policy.critic.action_dim)
        prefix_onehot, prefix_mask = self._full_prefix_inputs(buffer, env_indices)
        eq, factors = self.policy.evaluate_gpae_sequence(share_obs,
                                                         rnn_states_critic,
                                                         seq_masks,
                                                         own_index,
                                                         own_actions,
                                                         own_probs,
                                                         prefix_onehot,
                                                         prefix_mask,
                                                         self.data_chunk_length)
        n_envs = buffer.n_rollout_threads if env_indices is None else len(env_indices)
        eq = eq.view(buffer.episode_length, n_envs, buffer.num_agents, 1)
        factors = factors.view_as(eq)

        # per-agent GPAE recursion in raw reward scale (as the DAE recursion);
        # E_Q at the rollout boundary s_T is approximated by V(s_T), exact in
        # expectation over a_-k and only touching the truncation boundary.
        if self.value_normalizer is not None:
            eq_denorm = check(self.value_normalizer.denormalize(eq.detach())).to(**self.tpdv)
        else:
            eq_denorm = eq.detach()

        with torch.no_grad():
            if rho_own is not None:
                # DT-ISR trace (GPAE paper Eq. 8): sensitive to the own ratio,
                # robust to the others' joint ratio through the eta cap.
                rho_bar = torch.clamp(rho_own, max=1.0)
                trace = self.dae_gpae_lambda * torch.clamp(
                    rho_own * torch.clamp(rho_others, max=self.dae_trace_eta), max=1.0)
            else:
                rho_bar = None
                trace = None
            gpae_adv = []
            adv = torch.zeros_like(bootstrap)
            eq_next = bootstrap
            for step in reversed(range(buffer.episode_length)):
                delta = rewards[step] + self.gamma * masks[step + 1] * eq_next - eq_denorm[step]
                if trace is None:
                    c_next = self.dae_gpae_lambda
                else:
                    # the product in Eq. 7 starts at t + 1; the tail beyond the
                    # rollout is zero so the boundary trace value is irrelevant
                    c_next = trace[step + 1] if step + 1 < buffer.episode_length else 0.0
                adv = delta + self.gamma * masks[step + 1] * c_next * adv
                adv = adv * bad_masks[step + 1]
                gpae_adv.insert(0, adv)
                eq_next = eq_denorm[step]
            gpae_adv = torch.stack(gpae_adv, dim=0)

        eq_targets = eq_denorm + (gpae_adv if rho_bar is None else rho_bar * gpae_adv)
        if self.value_normalizer is not None:
            eq_error = check(self.value_normalizer.normalize(eq_targets)).to(**self.tpdv) - eq
        else:
            eq_error = eq_targets - eq
        eq_loss = mse_loss(eq_error)

        # the factor head lives in normalized units (see _evaluate_dae_buffer)
        adv_std = self._value_norm_std()
        target_norm = gpae_adv / adv_std if adv_std is not None else gpae_adv
        aux_loss = mse_loss(target_norm - factors)

        aux_weights = active_masks if rho_bar is None else active_masks * rho_bar
        if self._use_value_active_masks:
            eq_loss = (eq_loss * active_masks).sum() / active_masks.sum()
            aux_loss = (aux_loss * aux_weights).sum() / (aux_weights.sum() + 1e-8)
        else:
            eq_loss = eq_loss.mean()
            if rho_bar is None:
                aux_loss = aux_loss.mean()
            else:
                aux_loss = (aux_loss * rho_bar).sum() / (rho_bar.sum() + 1e-8)

        with torch.no_grad():
            active = (active_masks != 0.0).reshape(-1)
            f = factors.reshape(-1)[active]
            t = target_norm.reshape(-1)[active]
            if f.numel() > 1 and f.std() > 1e-8 and t.std() > 1e-8:
                corr = ((f - f.mean()) * (t - t.mean())).mean() / (f.std() * t.std())
                self._last_gpae_factor_corr = float(corr)
            else:
                self._last_gpae_factor_corr = 0.0
            self._last_gpae_adv_rms = float(gpae_adv.reshape(-1)[active].pow(2).mean().sqrt())
            self._last_gpae_eq_loss = float(eq_loss)
            self._last_gpae_aux_loss = float(aux_loss)

        return eq_loss, aux_loss

    def _dae_update_minibatch(self, buffer, env_indices=None):
        values, advantages = self._evaluate_dae_buffer(buffer, env_indices)
        if self._dae_factor_mode:
            # the residual recursion consumes the joint advantage: sum the
            # per-agent factors and broadcast back over the agent dimension
            advantages = advantages.sum(dim=2, keepdim=True).expand_as(values)
        rewards = check(self._env_slice(buffer.rewards, env_indices, include_last=True)).to(**self.tpdv)
        masks = check(self._env_slice(buffer.masks, env_indices, include_last=True)).to(**self.tpdv)
        bad_masks = check(self._env_slice(buffer.bad_masks, env_indices, include_last=True)).to(**self.tpdv)
        active_masks = check(self._env_slice(buffer.active_masks, env_indices, include_last=False)).to(**self.tpdv)
        next_value = getattr(buffer, "next_value", buffer.value_preds[-1])
        if env_indices is not None:
            next_value = next_value[env_indices]

        # The recursion runs in raw reward scale; critic outputs live in
        # normalized space (as in cal_value_loss), so bootstrap values are
        # denormalized before entering the recursion.
        if self.value_normalizer is not None:
            bootstrap = check(self.value_normalizer.denormalize(check(next_value))).to(**self.tpdv)
            values_denorm = check(self.value_normalizer.denormalize(values.detach())).to(**self.tpdv)
        else:
            bootstrap = check(next_value).to(**self.tpdv)
            values_denorm = values.detach()

        target = bootstrap
        targets = []
        for step in reversed(range(buffer.episode_length)):
            target = rewards[step] - advantages[step] + self.gamma * masks[step + 1] * target
            target = target * bad_masks[step + 1] + (1.0 - bad_masks[step + 1]) * values_denorm[step]
            targets.insert(0, target)

        targets = torch.stack(targets, dim=0)
        if self.value_normalizer is not None:
            # Update normalizer stats with advantage-free returns: transformed
            # returns share their mean with plain returns (centering invariance)
            # but feeding them back into the stats couples sigma to the advantage
            # head's own scale and diverges.
            with torch.no_grad():
                plain = bootstrap
                plain_returns = []
                for step in reversed(range(buffer.episode_length)):
                    plain = rewards[step] + self.gamma * masks[step + 1] * plain
                    plain = plain * bad_masks[step + 1] + (1.0 - bad_masks[step + 1]) * values_denorm[step]
                    plain_returns.insert(0, plain)
                plain_returns = torch.stack(plain_returns, dim=0)
            self.value_normalizer.update(plain_returns.reshape(-1, 1))
            error = self.value_normalizer.normalize(targets) - values
        else:
            error = targets - values
        dae_loss = mse_loss(error)

        with torch.no_grad():
            normalized_targets = error + values
            target_var = normalized_targets.var()
            self._last_dae_explained_var = float(1.0 - error.var() / (target_var + 1e-8))

        if self._use_value_active_masks:
            dae_loss = (dae_loss * active_masks).sum() / active_masks.sum()
        else:
            dae_loss = dae_loss.mean()

        total_loss = dae_loss * self.dae_loss_coef
        if self._use_gpae_aux:
            eq_loss, aux_loss = self._gpae_losses(buffer, env_indices, rewards, masks,
                                                  bad_masks, active_masks, bootstrap)
            total_loss = total_loss + eq_loss + self.dae_gpae_coef * aux_loss

        self.policy.critic_optimizer.zero_grad()
        total_loss.backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()
        return dae_loss, critic_grad_norm

    def dae_update(self, buffer):
        if self.dae_num_mini_batch <= 1 or buffer.n_rollout_threads <= 1:
            result = self._dae_update_minibatch(buffer)
            self._first_dae_explained_var = self._last_dae_explained_var
            return result

        indices = np.random.permutation(buffer.n_rollout_threads)
        mini_batches = np.array_split(indices, min(self.dae_num_mini_batch, buffer.n_rollout_threads))
        losses, grad_norms = [], []
        first_ev = None
        for env_indices in mini_batches:
            if len(env_indices) == 0:
                continue
            dae_loss, critic_grad_norm = self._dae_update_minibatch(buffer, env_indices)
            if first_ev is None:
                first_ev = self._last_dae_explained_var
            losses.append(dae_loss.item())
            grad_norms.append(critic_grad_norm.item() if torch.is_tensor(critic_grad_norm) else float(critic_grad_norm))
        self._first_dae_explained_var = first_ev

        mean_loss = torch.tensor(np.mean(losses), **self.tpdv)
        mean_grad_norm = torch.tensor(np.mean(grad_norms), **self.tpdv)
        return mean_loss, mean_grad_norm

    # ---- off-policy replay of the DAE / GPAE head training (actor stays on-policy) ----

    def _make_dae_snapshot(self, buffer):
        """
        Copy the arrays needed to retrain the heads on this batch later. Stored
        per-step actor hidden states would be stale under future policies, so
        only the sequence-start states are kept and probabilities are recomputed
        with the current actor at replay time. action_log_probs record the
        behavior policy mu for the importance ratios.
        """
        snap = types.SimpleNamespace()
        snap.share_obs = buffer.share_obs.copy()
        snap.obs = buffer.obs.copy()
        snap.actions = buffer.actions.copy()
        snap.action_log_probs = buffer.action_log_probs.copy()
        snap.rewards = buffer.rewards.copy()
        snap.masks = buffer.masks.copy()
        snap.bad_masks = buffer.bad_masks.copy()
        snap.active_masks = buffer.active_masks.copy()
        snap.available_actions = (None if buffer.available_actions is None
                                  else buffer.available_actions.copy())
        snap.rnn_states = buffer.rnn_states[0:1].copy()
        snap.rnn_states_critic = buffer.rnn_states_critic[0:1].copy()
        snap.next_value = np.array(getattr(buffer, "next_value", buffer.value_preds[-1]), copy=True)
        snap.episode_length = buffer.episode_length
        snap.num_agents = buffer.num_agents
        snap.n_rollout_threads = buffer.n_rollout_threads
        snap.birth = self._dae_train_calls
        return snap

    @torch.no_grad()
    def _policy_probs_replay(self, snapshot, env_indices=None):
        """Current-policy action probabilities over a replayed batch, rebuilding
        actor hidden states from the sequence start (truncated BPTT)."""
        obs = self._env_slice(snapshot.obs, env_indices, include_last=False).reshape(
            -1, *snapshot.obs.shape[3:])
        rnn_states = snapshot.rnn_states[0]
        if env_indices is not None:
            rnn_states = rnn_states[env_indices]
        rnn_states = rnn_states.reshape(-1, *snapshot.rnn_states.shape[3:])
        masks = self._env_slice(snapshot.masks, env_indices, include_last=False).reshape(-1, 1)
        if snapshot.available_actions is None:
            available_actions = None
        else:
            available_actions = self._env_slice(snapshot.available_actions, env_indices,
                                                include_last=False).reshape(
                -1, snapshot.available_actions.shape[-1])

        probs = self.policy.get_action_probs_sequence(obs, rnn_states, masks,
                                                      available_actions, self.data_chunk_length)
        n_envs = snapshot.n_rollout_threads if env_indices is None else len(env_indices)
        return probs.view(snapshot.episode_length, n_envs, snapshot.num_agents, -1).detach()

    def _replay_ratios(self, snapshot, env_indices, policy_probs):
        """
        Per-agent importance ratios rho^k = pi_k(a_k) / mu_k(a_k) plus the
        others' and joint products, all (T, n_envs, num_agents, 1). Products run
        in log space; the clamp keeps degenerate rows (e.g. dead agents) benign.
        """
        actions = self._env_slice(snapshot.actions, env_indices, include_last=True)
        act_idx = torch.from_numpy(actions[..., 0].astype(np.int64)).to(policy_probs.device)
        act_idx = act_idx.clamp(0, policy_probs.shape[-1] - 1)
        pi = policy_probs.gather(-1, act_idx.unsqueeze(-1))
        mu_log = check(self._env_slice(snapshot.action_log_probs, env_indices,
                                       include_last=True)).to(**self.tpdv)
        log_rho = torch.clamp(torch.log(pi + 1e-10) - mu_log, min=-20.0, max=20.0)
        rho_own = torch.exp(log_rho)
        log_joint = log_rho.sum(dim=2, keepdim=True)
        rho_others = torch.exp(torch.clamp(log_joint - log_rho, min=-20.0, max=20.0))
        rho_joint = torch.exp(torch.clamp(log_joint, min=-20.0, max=20.0)).expand_as(rho_own)
        return rho_own, rho_others, rho_joint

    def _dae_replay_minibatch(self, snapshot, env_indices=None):
        """
        One head update on replayed data. The residual recursion becomes V-trace
        on the difference-reward MDP: with the truncated joint ratio
        c = min(1, rho_joint),
          v_t = V_t + c_t * delta_t + gamma * m_{t+1} * c_t * (v_{t+1} - V_{t+1}),
          delta_t = r_t - A_t + gamma * m_{t+1} * V_{t+1} - V_t,
        which reduces EXACTLY to the on-policy recursion when rho = 1. The GPAE
        losses use per-agent DT-ISR traces (see _gpae_losses). The value
        normalizer statistics are NOT updated from replayed returns.
        """
        policy_probs = self._policy_probs_replay(snapshot, env_indices)
        values, advantages = self._evaluate_dae_buffer(snapshot, env_indices,
                                                       policy_probs=policy_probs)
        if self._dae_factor_mode:
            advantages = advantages.sum(dim=2, keepdim=True).expand_as(values)
        rewards = check(self._env_slice(snapshot.rewards, env_indices, include_last=True)).to(**self.tpdv)
        masks = check(self._env_slice(snapshot.masks, env_indices, include_last=True)).to(**self.tpdv)
        bad_masks = check(self._env_slice(snapshot.bad_masks, env_indices, include_last=True)).to(**self.tpdv)
        active_masks = check(self._env_slice(snapshot.active_masks, env_indices,
                                             include_last=False)).to(**self.tpdv)
        next_value = snapshot.next_value
        if env_indices is not None:
            next_value = next_value[env_indices]

        if self.value_normalizer is not None:
            bootstrap = check(self.value_normalizer.denormalize(check(next_value))).to(**self.tpdv)
            values_denorm = check(self.value_normalizer.denormalize(values.detach())).to(**self.tpdv)
        else:
            bootstrap = check(next_value).to(**self.tpdv)
            values_denorm = values.detach()

        rho_own, rho_others, rho_joint = self._replay_ratios(snapshot, env_indices, policy_probs)
        c_joint = torch.clamp(rho_joint, max=1.0)

        episode_length = snapshot.episode_length
        target_next = bootstrap
        targets = []
        for step in reversed(range(episode_length)):
            v_next = values_denorm[step + 1] if step + 1 < episode_length else bootstrap
            delta = (rewards[step] - advantages[step]
                     + self.gamma * masks[step + 1] * v_next - values_denorm[step])
            target = (values_denorm[step] + c_joint[step] * delta
                      + self.gamma * masks[step + 1] * c_joint[step] * (target_next - v_next))
            target = target * bad_masks[step + 1] + (1.0 - bad_masks[step + 1]) * values_denorm[step]
            targets.insert(0, target)
            target_next = target
        targets = torch.stack(targets, dim=0)

        if self.value_normalizer is not None:
            error = self.value_normalizer.normalize(targets) - values
        else:
            error = targets - values
        replay_loss = mse_loss(error)
        if self._use_value_active_masks:
            replay_loss = (replay_loss * active_masks).sum() / active_masks.sum()
        else:
            replay_loss = replay_loss.mean()

        total_loss = replay_loss * self.dae_loss_coef
        if self._use_gpae_aux:
            eq_loss, aux_loss = self._gpae_losses(snapshot, env_indices, rewards, masks,
                                                  bad_masks, active_masks, bootstrap,
                                                  policy_probs=policy_probs,
                                                  rho_own=rho_own, rho_others=rho_others)
            total_loss = total_loss + eq_loss + self.dae_gpae_coef * aux_loss

        self.policy.critic_optimizer.zero_grad()
        total_loss.backward()
        if self._use_max_grad_norm:
            nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        self.policy.critic_optimizer.step()

        with torch.no_grad():
            active = (active_masks != 0.0)
            self._last_dae_replay_loss = float(replay_loss)
            self._last_replay_c_joint_mean = float(c_joint[active].mean())
            self._last_replay_rho_own_mean = float(rho_own[active].mean())
            # age-0 self-check: within the same train() call the actor has not
            # been updated yet, so pi == mu must hold exactly. rho at t = 0 uses
            # the STORED start-of-window hidden state directly, so a t = 0
            # deviation means the prob evaluation itself is inconsistent, while
            # t = 0 OK but t > 0 off means the RNN sequence rebuild diverges
            # from the rollout hidden states.
            if self._dae_train_calls - getattr(snapshot, "birth", -1) == 0:
                rho_t0 = float(rho_own[0][active[0]].mean())
                rho_rest = float(rho_own[1:][active[1:]].mean())
                self._last_replay_age0_rho_t0 = rho_t0
                self._last_replay_age0_rho_rest = rho_rest
                if abs(rho_t0 - 1.0) > 1e-3 or abs(rho_rest - 1.0) > 1e-3:
                    print("[DAE replay WARNING] age-0 snapshot but rho != 1: "
                          "rho(t=0)={:.4f}, rho(t>0)={:.4f} -- replayed pi does not "
                          "reproduce the rollout policy; importance weighting is "
                          "unreliable.".format(rho_t0, rho_rest))
        return replay_loss

    def dae_replay_update(self):
        """One pass over a snapshot sampled uniformly from the replay store."""
        snapshot = self._dae_replay[np.random.randint(len(self._dae_replay))]
        if self.dae_num_mini_batch <= 1 or snapshot.n_rollout_threads <= 1:
            loss = self._dae_replay_minibatch(snapshot)
            rho_means = [self._last_replay_rho_own_mean]
        else:
            indices = np.random.permutation(snapshot.n_rollout_threads)
            mini_batches = np.array_split(indices,
                                          min(self.dae_num_mini_batch, snapshot.n_rollout_threads))
            losses, rho_means = [], []
            for env_indices in mini_batches:
                if len(env_indices) == 0:
                    continue
                losses.append(self._dae_replay_minibatch(snapshot, env_indices).item())
                rho_means.append(self._last_replay_rho_own_mean)
            loss = torch.tensor(np.mean(losses), **self.tpdv)
        # per-age ratio diagnostic: staleness of each snapshot generation
        age = self._dae_train_calls - snapshot.birth
        if not hasattr(self, "_replay_rho_by_age"):
            self._replay_rho_by_age = {}
        self._replay_rho_by_age[age] = float(np.mean(rho_means))
        return loss

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                        self.clip_param)
        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, sample, update_actor=True):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        if len(sample) == 12:
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
            value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            adv_targ, available_actions_batch = sample
        else:
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
            value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            adv_targ, available_actions_batch, _ = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(share_obs_batch,
                                                                              obs_batch, 
                                                                              rnn_states_batch, 
                                                                              rnn_states_critic_batch, 
                                                                              actions_batch, 
                                                                              masks_batch, 
                                                                              available_actions_batch,
                                                                              active_masks_batch)
        # actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2),
                                             dim=-1,
                                             keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss

        self.policy.actor_optimizer.zero_grad()

        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()

        if self._use_dae:
            value_loss = torch.zeros((), **self.tpdv)
            critic_grad_norm = torch.zeros((), **self.tpdv)
        else:
            # critic update
            value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

            self.policy.critic_optimizer.zero_grad()

            (value_loss * self.value_loss_coef).backward()

            if self._use_max_grad_norm:
                critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
            else:
                critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

            self.policy.critic_optimizer.step()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def train(self, buffer, update_actor=True):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        if self._use_dae:
            dae_value_loss = 0
            dae_critic_grad_norm = 0
            dae_ev_fresh = 0.0
            for epoch_i in range(self.dae_epoch):
                value_loss, critic_grad_norm = self.dae_update(buffer)
                if epoch_i == 0:
                    # explained variance of the previous iteration's network on
                    # fresh data, before any update this iteration
                    dae_ev_fresh = getattr(self, "_first_dae_explained_var", 0.0)
                dae_value_loss += value_loss.item()
                dae_critic_grad_norm += critic_grad_norm.item() if torch.is_tensor(critic_grad_norm) else float(critic_grad_norm)

            # off-policy head refinement on replayed snapshots (before the actor
            # advantages are computed, and before any actor update, so fresh-data
            # centering stays exact)
            if self._use_dae_replay:
                self._dae_replay.append(self._make_dae_snapshot(buffer))
                replay_losses = []
                for _ in range(self.dae_replay_updates):
                    replay_losses.append(float(self.dae_replay_update()))
                self._dae_replay_loss_mean = float(np.mean(replay_losses))

            advantages = self.compute_dae_advantages(buffer)
            dae_value_loss /= max(self.dae_epoch, 1)
            dae_critic_grad_norm /= max(self.dae_epoch, 1)

            # GAE -> DAE warmup: blend the actor advantage from GAE (accurate but
            # noisy early on) toward DAE as the head becomes trustworthy, to avoid
            # the policy locking onto persistent-but-wrong head preferences before
            # the head has learned (premature entropy collapse).
            if self.dae_warmup_updates > 0:
                dae_mix_weight = min(1.0, self._dae_train_calls / self.dae_warmup_updates)
            else:
                dae_mix_weight = 1.0
            self._dae_train_calls += 1
            dae_pure_advantages = advantages
            if dae_mix_weight < 1.0:
                gae_advantages = self._reference_gae_advantages(buffer)
                advantages = dae_mix_weight * advantages + (1.0 - dae_mix_weight) * gae_advantages
            self.dae_mix_weight = dae_mix_weight
        elif self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]

        if not self._use_dae:
            advantages_copy = advantages.copy()
            advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
        

        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0
        if self._use_dae:
            # diagnostics are computed on the pure DAE signal, not the warmup blend
            train_info['dae_adv_mean'] = float(np.nanmean(np.where(buffer.active_masks[:-1] == 0.0, np.nan, dae_pure_advantages)))
            train_info['dae_adv_std'] = float(np.nanstd(np.where(buffer.active_masks[:-1] == 0.0, np.nan, dae_pure_advantages)))
            train_info['dae_adv_raw_mean'] = self.dae_adv_raw_mean
            train_info['dae_adv_raw_std'] = self.dae_adv_raw_std
            train_info['dae_adv_raw_rms'] = self.dae_adv_raw_rms
            gae_advantages = self._reference_gae_advantages(buffer)
            # GAE alignment is only comparable against the joint signal; with
            # agent-wise factors, compare their sum instead of the raw factors.
            if self._dae_factor_mode and self.dae_actor_adv == "agent":
                align_signal = np.broadcast_to(dae_pure_advantages.sum(axis=2, keepdims=True),
                                               dae_pure_advantages.shape)
            else:
                align_signal = dae_pure_advantages
            dae_gae_corr, dae_gae_sign, dae_gae_cos = self._advantage_alignment_stats(
                align_signal, gae_advantages, buffer.active_masks[:-1])
            train_info['dae_gae_corr'] = dae_gae_corr
            train_info['dae_gae_sign'] = dae_gae_sign
            train_info['dae_gae_cos'] = dae_gae_cos
            train_info['dae_explained_var'] = getattr(self, "_last_dae_explained_var", 0.0)
            train_info['dae_explained_var_fresh'] = dae_ev_fresh
            train_info['dae_mix_weight'] = self.dae_mix_weight
            if self._dae_factor_mode:
                train_info['dae_factor_rms_spread'] = self.dae_factor_rms_spread
            if self._use_gpae_aux:
                train_info['gpae_eq_loss'] = getattr(self, "_last_gpae_eq_loss", 0.0)
                train_info['gpae_aux_loss'] = getattr(self, "_last_gpae_aux_loss", 0.0)
                train_info['gpae_adv_rms'] = getattr(self, "_last_gpae_adv_rms", 0.0)
                train_info['gpae_factor_corr'] = getattr(self, "_last_gpae_factor_corr", 0.0)
            if self._use_dae_replay:
                train_info['dae_replay_loss'] = getattr(self, "_dae_replay_loss_mean", 0.0)
                train_info['dae_replay_c_joint'] = getattr(self, "_last_replay_c_joint_mean", 0.0)
                train_info['dae_replay_rho_own'] = getattr(self, "_last_replay_rho_own_mean", 0.0)
                for age, rho in sorted(getattr(self, "_replay_rho_by_age", {}).items()):
                    train_info['dae_replay_rho_own_age{}'.format(age)] = rho
                self._replay_rho_by_age = {}
                if hasattr(self, "_last_replay_age0_rho_t0"):
                    train_info['dae_replay_age0_rho_t0'] = self._last_replay_age0_rho_t0
                    train_info['dae_replay_age0_rho_rest'] = self._last_replay_age0_rho_rest
                    # fresh-only: do not re-log stale values on iterations where
                    # no age-0 snapshot was sampled
                    del self._last_replay_age0_rho_t0
                    del self._last_replay_age0_rho_rest

        anomaly_masks = getattr(buffer, "anomaly_masks", None)
        if self.anomaly_agent_id >= 0 and anomaly_masks is not None:
            # measured on the pure (unblended) per-agent signal the actor would see
            gap_signal = dae_pure_advantages if self._use_dae else advantages
            gap, count = self._anomaly_advantage_gap(gap_signal, anomaly_masks,
                                                     buffer.active_masks[:-1])
            train_info['anomaly_adv_gap'] = gap
            train_info['anomaly_count'] = count

        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:

                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights \
                    = self.ppo_update(sample, update_actor)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.num_mini_batch

        # Only the PPO-loop accumulators are epoch SUMS that need averaging.
        # Diagnostics assigned once above must not be divided (dividing them was
        # a long-standing bug that silently scaled every diagnostic, e.g.
        # importance ratios reading 0.2 instead of 1.0 with ppo_epoch 5).
        for k in ("value_loss", "policy_loss", "dist_entropy",
                  "actor_grad_norm", "critic_grad_norm", "ratio"):
            train_info[k] /= num_updates

        if self._use_dae:
            train_info['value_loss'] = dae_value_loss
            train_info['critic_grad_norm'] = dae_critic_grad_norm
 
        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
