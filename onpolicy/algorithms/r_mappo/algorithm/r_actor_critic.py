import torch
import torch.nn as nn
from onpolicy.algorithms.utils.util import init, check
from onpolicy.algorithms.utils.cnn import CNNBase
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.act import ACTLayer
from onpolicy.algorithms.utils.popart import PopArt
from onpolicy.utils.util import get_shape_from_obs_space


class R_Actor(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(R_Actor, self).__init__()
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(args, obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain, args)

        self.to(device)
        self.algo = args.algorithm_name

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)

        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param rnn_states: (torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if self.algo == "hatrpo":
            action_log_probs, dist_entropy ,action_mu, action_std, all_probs= self.act.evaluate_actions_trpo(actor_features,
                                                                    action, available_actions,
                                                                    active_masks=
                                                                    active_masks if self._use_policy_active_masks
                                                                    else None)

            return action_log_probs, dist_entropy, action_mu, action_std, all_probs
        else:
            action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features,
                                                                    action, available_actions,
                                                                    active_masks=
                                                                    active_masks if self._use_policy_active_masks
                                                                    else None)

        return action_log_probs, dist_entropy

    def get_action_probs(self, obs, rnn_states, masks, available_actions=None):
        """
        Compute action probabilities for all discrete actions.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, _ = self.rnn(actor_features, rnn_states, masks)

        return self.act.get_probs(actor_features, available_actions)

    def get_action_probs_sequence(self, obs, rnn_states, masks, available_actions=None,
                                  chunk_length=None):
        """
        Recompute action probabilities over full time-major sequences with the
        CURRENT actor, rebuilding hidden states from the sequence start (needed
        for replayed data, whose stored per-step hidden states are stale).
        :param obs: (T * B, obs_dim) time-major flattened sequence.
        :param rnn_states: (B, recurrent_N, hidden_size) hidden states at the sequence start.
        :param masks: (T * B, 1) mask tensor; zeros reset hidden states at episode boundaries.
        :param chunk_length: (int) BPTT truncation length (hidden states detached between chunks).
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            batch = rnn_states.size(0)
            steps = actor_features.size(0) // batch
            if chunk_length is None or chunk_length >= steps:
                actor_features, _ = self.rnn(actor_features, rnn_states, masks)
            else:
                features = actor_features.view(steps, batch, -1)
                seq_masks = masks.view(steps, batch, -1)
                hxs = rnn_states
                outputs = []
                for start in range(0, steps, chunk_length):
                    end = min(start + chunk_length, steps)
                    chunk_features, hxs = self.rnn(
                        features[start:end].reshape((end - start) * batch, -1),
                        hxs,
                        seq_masks[start:end].reshape((end - start) * batch, -1))
                    hxs = hxs.detach()
                    outputs.append(chunk_features)
                actor_features = torch.cat(outputs, dim=0).view(steps * batch, -1)

        return self.act.get_probs(actor_features, available_actions)


class R_Critic(nn.Module):
    """
    Critic network class for MAPPO. Outputs value function predictions given centralized input (MAPPO) or
                            local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, cent_obs_space, action_space=None, num_agents=1, device=torch.device("cpu")):
        super(R_Critic, self).__init__()
        if isinstance(action_space, (torch.device, str)):
            device = action_space
            action_space = None
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self._use_dae = getattr(args, "use_dae", False)
        self.num_agents = num_agents
        self.action_space = action_space
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        base = CNNBase if len(cent_obs_shape) == 3 else MLPBase
        self.base = base(args, cent_obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))

        if self._use_dae:
            if action_space is None:
                raise ValueError("DAE critic requires action_space.")
            if action_space.__class__.__name__ != "Discrete":
                raise NotImplementedError("Initial DAE-MAPPO implementation supports Discrete action spaces.")
            self.action_dim = action_space.n
            self.dae_head_mode = getattr(args, "dae_head", "additive")
            # ordered head conditions each agent's factor on preceding agents'
            # actions: input gains one-hot slots for all agents (zero for
            # non-predecessors) plus a predecessor mask.
            adv_in_size = self.hidden_size
            if self.dae_head_mode == "ordered":
                adv_in_size += self.num_agents * self.action_dim + self.num_agents
            # output-layer init gain: 1.0 reproduces the strong 7/14 baseline run;
            # 0.1 is the official-DAE-style small init (see --dae_head_gain)
            dae_head_gain = getattr(args, "dae_head_gain", 1.0)
            dae_head_hidden_size = getattr(args, "dae_head_hidden_size", 0)
            if dae_head_hidden_size > 0:
                use_relu = args.use_ReLU
                active_fn = nn.ReLU() if use_relu else nn.Tanh()
                hidden_gain = nn.init.calculate_gain('relu' if use_relu else 'tanh')
                self.adv_out = nn.Sequential(
                    init(nn.Linear(adv_in_size, dae_head_hidden_size),
                         init_method, lambda x: nn.init.constant_(x, 0), gain=hidden_gain),
                    active_fn,
                    nn.LayerNorm(dae_head_hidden_size),
                    init(nn.Linear(dae_head_hidden_size, self.num_agents * self.action_dim),
                         init_method, lambda x: nn.init.constant_(x, 0), gain=dae_head_gain),
                )
            else:
                self.adv_out = init(nn.Linear(adv_in_size, self.num_agents * self.action_dim),
                                    init_method, lambda x: nn.init.constant_(x, 0), gain=dae_head_gain)

            # GPAE auxiliary: dedicated counterfactual-value head E_Q^k(s, a_-k)
            # (own-action-marginalized joint value). Separate parameters from
            # adv_out so its reward-grounded target is not corrupted by the
            # factor head's own decomposition errors; same input layout as the
            # ordered head with a full prefix (all other agents' actions).
            self.use_gpae_head = getattr(args, "dae_gpae_coef", 0.0) > 0.0
            if self.use_gpae_head:
                assert self.dae_head_mode == "ordered", \
                    "the GPAE auxiliary loss requires dae_head=ordered."
                eq_in_size = self.hidden_size + self.num_agents * self.action_dim + self.num_agents
                if dae_head_hidden_size > 0:
                    use_relu = args.use_ReLU
                    active_fn = nn.ReLU() if use_relu else nn.Tanh()
                    hidden_gain = nn.init.calculate_gain('relu' if use_relu else 'tanh')
                    self.eq_out = nn.Sequential(
                        init(nn.Linear(eq_in_size, dae_head_hidden_size),
                             init_method, lambda x: nn.init.constant_(x, 0), gain=hidden_gain),
                        active_fn,
                        nn.LayerNorm(dae_head_hidden_size),
                        init(nn.Linear(dae_head_hidden_size, self.num_agents),
                             init_method, lambda x: nn.init.constant_(x, 0)),
                    )
                else:
                    self.eq_out = init(nn.Linear(eq_in_size, self.num_agents),
                                       init_method, lambda x: nn.init.constant_(x, 0))

        self.to(device)

    def _features(self, cent_obs, rnn_states, masks):
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(cent_obs)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        return critic_features, rnn_states

    def _advantage_table(self, critic_features, prefix_onehot=None, prefix_mask=None):
        if self.dae_head_mode == "ordered":
            if prefix_onehot is None or prefix_mask is None:
                raise RuntimeError("ordered DAE head requires prefix_onehot and prefix_mask.")
            prefix_onehot = check(prefix_onehot).to(**self.tpdv)
            prefix_mask = check(prefix_mask).to(**self.tpdv)
            critic_features = torch.cat([critic_features, prefix_onehot, prefix_mask], dim=-1)
        advantages = self.adv_out(critic_features)
        return advantages.view(-1, self.num_agents, self.action_dim)

    def _joint_action_indices(self, joint_actions):
        joint_actions = check(joint_actions).to(device=self.tpdv["device"])
        joint_actions = joint_actions.long().view(joint_actions.shape[0], self.num_agents, -1)
        joint_actions = joint_actions[..., 0].clamp(min=0, max=self.action_dim - 1)
        return joint_actions

    def _gather_joint_advantages(self, advantage_table, joint_actions):
        joint_actions = self._joint_action_indices(joint_actions)
        advantages = advantage_table.gather(dim=-1, index=joint_actions.unsqueeze(-1))
        return advantages.sum(dim=1)

    def forward(self, cent_obs, rnn_states, masks):
        """
        Compute actions from the given inputs.
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        critic_features, rnn_states = self._features(cent_obs, rnn_states, masks)
        values = self.v_out(critic_features)

        return values, rnn_states

    def evaluate_dae(self, cent_obs, rnn_states, masks, joint_actions):
        if not self._use_dae:
            raise RuntimeError("evaluate_dae called when DAE is disabled.")

        critic_features, rnn_states = self._features(cent_obs, rnn_states, masks)
        values = self.v_out(critic_features)
        advantage_table = self._advantage_table(critic_features)
        advantages = self._gather_joint_advantages(advantage_table, joint_actions)

        return values, advantages, rnn_states

    def evaluate_centered_dae(self, cent_obs, rnn_states, masks, joint_actions, joint_action_probs):
        if not self._use_dae:
            raise RuntimeError("evaluate_centered_dae called when DAE is disabled.")

        critic_features, rnn_states = self._features(cent_obs, rnn_states, masks)
        joint_action_probs = check(joint_action_probs).to(**self.tpdv)
        joint_action_probs = joint_action_probs.view(joint_action_probs.shape[0],
                                                     self.num_agents,
                                                     self.action_dim)

        values = self.v_out(critic_features)
        advantage_table = self._advantage_table(critic_features)
        expected_advantages = (joint_action_probs * advantage_table).sum(dim=-1, keepdim=True)
        centered_advantage_table = advantage_table - expected_advantages
        advantages = self._gather_joint_advantages(centered_advantage_table, joint_actions)

        return values, advantages, rnn_states

    def _sequence_features(self, cent_obs, rnn_states, masks, chunk_length=None):
        """
        Run the critic over full time-major sequences, recomputing RNN hidden
        states in the forward pass with truncated BPTT.
        :param cent_obs: (T * B, obs_dim) time-major flattened sequence.
        :param rnn_states: (B, recurrent_N, hidden_size) hidden states at the start of the sequence.
        :param masks: (T * B, 1) mask tensor; zeros reset the hidden state at episode boundaries.
        :param chunk_length: (int) BPTT truncation length; hidden states are detached between chunks.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(cent_obs)
        if not (self._use_naive_recurrent_policy or self._use_recurrent_policy):
            return critic_features

        batch = rnn_states.size(0)
        steps = critic_features.size(0) // batch
        if chunk_length is None or chunk_length >= steps:
            critic_features, _ = self.rnn(critic_features, rnn_states, masks)
            return critic_features

        features = critic_features.view(steps, batch, -1)
        masks = masks.view(steps, batch, -1)
        hxs = rnn_states
        outputs = []
        for start in range(0, steps, chunk_length):
            end = min(start + chunk_length, steps)
            chunk_features, hxs = self.rnn(features[start:end].reshape((end - start) * batch, -1),
                                           hxs,
                                           masks[start:end].reshape((end - start) * batch, -1))
            hxs = hxs.detach()
            outputs.append(chunk_features)
        return torch.cat(outputs, dim=0).view(steps * batch, -1)

    def evaluate_dae_sequence(self, cent_obs, rnn_states, masks, joint_actions,
                              joint_action_probs=None, chunk_length=None):
        """
        Evaluate values and (optionally centered) DAE advantages over full
        time-major sequences with truncated BPTT, so the recurrent critic is
        trained through time instead of from stale per-step rollout states.
        """
        if not self._use_dae:
            raise RuntimeError("evaluate_dae_sequence called when DAE is disabled.")

        critic_features = self._sequence_features(cent_obs, rnn_states, masks, chunk_length)
        values = self.v_out(critic_features)
        advantage_table = self._advantage_table(critic_features)
        if joint_action_probs is not None:
            joint_action_probs = check(joint_action_probs).to(**self.tpdv)
            joint_action_probs = joint_action_probs.view(joint_action_probs.shape[0],
                                                         self.num_agents,
                                                         self.action_dim)
            expected_advantages = (joint_action_probs * advantage_table).sum(dim=-1, keepdim=True)
            advantage_table = advantage_table - expected_advantages
        advantages = self._gather_joint_advantages(advantage_table, joint_actions)

        return values, advantages

    def evaluate_factor_dae_sequence(self, cent_obs, rnn_states, masks, own_index, own_actions,
                                     own_action_probs=None, prefix_onehot=None, prefix_mask=None,
                                     chunk_length=None):
        """
        Evaluate values and each row's OWN per-agent advantage factor over full
        time-major sequences with truncated BPTT. Rows follow the buffer layout
        (T, envs, agents): row (t, b, k) yields agent k's centered factor
        A_k(s, a_<k, a_k) (ordered head) or A_k(s, a_k) (additive head).
        :param own_index: (T * B * N,) agent index of each row.
        :param own_actions: (T * B * N, 1) discrete action taken by the row's agent.
        :param own_action_probs: (T * B * N, action_dim) rollout-policy probabilities
                                 of the row's agent, for exact centering (None disables).
        :param prefix_onehot: (T * B * N, N * action_dim) predecessors' action one-hots
                              (zero slots for non-predecessors); ordered head only.
        :param prefix_mask: (T * B * N, N) predecessor indicator; ordered head only.
        """
        if not self._use_dae:
            raise RuntimeError("evaluate_factor_dae_sequence called when DAE is disabled.")

        critic_features = self._sequence_features(cent_obs, rnn_states, masks, chunk_length)
        values = self.v_out(critic_features)
        advantage_table = self._advantage_table(critic_features, prefix_onehot, prefix_mask)

        own_index = check(own_index).to(device=self.tpdv["device"]).long().view(-1, 1, 1)
        own_row = advantage_table.gather(dim=1, index=own_index.expand(-1, 1, self.action_dim)).squeeze(1)
        if own_action_probs is not None:
            own_action_probs = check(own_action_probs).to(**self.tpdv).view(-1, self.action_dim)
            own_row = own_row - (own_action_probs * own_row).sum(dim=-1, keepdim=True)

        own_actions = check(own_actions).to(device=self.tpdv["device"]).long().view(-1, 1)
        own_actions = own_actions.clamp(min=0, max=self.action_dim - 1)
        factors = own_row.gather(dim=-1, index=own_actions)

        return values, factors

    def evaluate_gpae_sequence(self, cent_obs, rnn_states, masks, own_index, own_actions,
                               own_action_probs, prefix_onehot, prefix_mask, chunk_length=None):
        """
        Evaluate the GPAE pair for each buffer row (t, b, k), with the FULL
        prefix (all agents j != k marked as predecessors):
        - eq: E_Q^k(s, a_-k) from the dedicated counterfactual-value head
          (normalized units, like v_out).
        - factors: the ordered head's fully conditioned centered factor
          A_k(s, a_-k, a_k) at the executed action (last-position factor).
        :param prefix_onehot: (T * B * N, N * action_dim) all OTHER agents' action
                              one-hots (full prefix).
        :param prefix_mask: (T * B * N, N) 1 - eye pattern marking all others.
        """
        if not (self._use_dae and self.dae_head_mode == "ordered" and self.use_gpae_head):
            raise RuntimeError("evaluate_gpae_sequence requires the ordered DAE head with the GPAE head enabled.")

        prefix_onehot = check(prefix_onehot).to(**self.tpdv)
        prefix_mask = check(prefix_mask).to(**self.tpdv)
        own_index = check(own_index).to(device=self.tpdv["device"]).long().view(-1, 1)

        critic_features = self._sequence_features(cent_obs, rnn_states, masks, chunk_length)
        advantage_table = self._advantage_table(critic_features, prefix_onehot, prefix_mask)
        own_row = advantage_table.gather(dim=1, index=own_index.view(-1, 1, 1).expand(-1, 1, self.action_dim)).squeeze(1)
        if own_action_probs is not None:
            own_action_probs = check(own_action_probs).to(**self.tpdv).view(-1, self.action_dim)
            own_row = own_row - (own_action_probs * own_row).sum(dim=-1, keepdim=True)
        own_actions = check(own_actions).to(device=self.tpdv["device"]).long().view(-1, 1)
        own_actions = own_actions.clamp(min=0, max=self.action_dim - 1)
        factors = own_row.gather(dim=-1, index=own_actions)

        eq = self.eq_out(torch.cat([critic_features, prefix_onehot, prefix_mask], dim=-1))
        eq = eq.gather(dim=-1, index=own_index)

        return eq, factors
