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
        joint_actions = self._joint_actions(buffer, env_indices)
        return share_obs, rnn_states_critic, masks, joint_actions

    def _value_norm_std(self):
        if self.value_normalizer is None:
            return None
        if hasattr(self.value_normalizer, "running_mean_var"):
            _, var = self.value_normalizer.running_mean_var()
        else:
            _, var = self.value_normalizer.debiased_mean_var()
        return torch.sqrt(var).reshape(1)

    def _evaluate_dae_buffer(self, buffer, env_indices=None):
        share_obs, rnn_states_critic, masks, joint_actions = self._dae_flat_inputs(buffer, env_indices)
        if self.dae_centering == "exact":
            # Centering probs use per-step stored actor hidden states; this is exact
            # only because DAE updates run before any actor update (actor == mu).
            joint_action_probs = self._joint_action_probs(buffer, env_indices)
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

    def _joint_action_probs(self, buffer, env_indices=None):
        policy_probs = self._policy_probs_buffer(buffer, env_indices).reshape(
            -1, buffer.num_agents, self.policy.critic.action_dim)
        return policy_probs.repeat_interleave(buffer.num_agents, dim=0)

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
    def compute_dae_advantages(self, buffer):
        if self.dae_num_mini_batch <= 1 or buffer.n_rollout_threads <= 1:
            _, advantages = self._evaluate_dae_buffer(buffer)
            advantages = advantages.cpu().numpy()
        else:
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

        advantages[buffer.active_masks[:-1] == 0.0] = 0.0
        raw_mean, raw_std, raw_rms = self._masked_advantage_stats(advantages, buffer.active_masks[:-1])
        self.dae_adv_raw_mean = raw_mean
        self.dae_adv_raw_std = raw_std
        self.dae_adv_raw_rms = raw_rms

        if self.dae_normalize_advantages:
            advantages = advantages / (raw_rms + 1e-5)

        return advantages

    def _dae_update_minibatch(self, buffer, env_indices=None):
        values, advantages = self._evaluate_dae_buffer(buffer, env_indices)
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

        self.policy.critic_optimizer.zero_grad()
        (dae_loss * self.dae_loss_coef).backward()

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
            dae_gae_corr, dae_gae_sign, dae_gae_cos = self._advantage_alignment_stats(
                dae_pure_advantages, gae_advantages, buffer.active_masks[:-1])
            train_info['dae_gae_corr'] = dae_gae_corr
            train_info['dae_gae_sign'] = dae_gae_sign
            train_info['dae_gae_cos'] = dae_gae_cos
            train_info['dae_explained_var'] = getattr(self, "_last_dae_explained_var", 0.0)
            train_info['dae_explained_var_fresh'] = dae_ev_fresh
            train_info['dae_mix_weight'] = self.dae_mix_weight

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

        for k in train_info.keys():
            if k in ("dae_adv_mean", "dae_adv_std", "dae_adv_raw_mean",
                     "dae_adv_raw_std", "dae_adv_raw_rms", "dae_gae_corr",
                     "dae_gae_sign", "dae_gae_cos", "dae_explained_var",
                     "dae_explained_var_fresh", "dae_mix_weight"):
                continue
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
