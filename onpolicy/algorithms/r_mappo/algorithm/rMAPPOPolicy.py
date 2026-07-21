import torch
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor, R_Critic
from onpolicy.utils.util import update_linear_schedule


class R_MAPPOPolicy:
    """
    MAPPO Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for MAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        self.actor = R_Actor(args, self.obs_space, self.act_space, self.device)
        self.critic = R_Critic(args, self.share_obs_space, self.act_space,
                               getattr(args, "num_agents", 1), self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
                    deterministic=False):
        """
        Compute actions and value function predictions for the given inputs.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
        :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """
        actions, action_log_probs, rnn_states_actor = self.actor(obs,
                                                                 rnn_states_actor,
                                                                 masks,
                                                                 available_actions,
                                                                 deterministic)

        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, rnn_states_critic, masks):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values

    def evaluate_dae(self, cent_obs, rnn_states_critic, masks, joint_actions):
        """
        Evaluate centralized value and joint-action DAE advantage.
        """
        values, advantages, _ = self.critic.evaluate_dae(cent_obs, rnn_states_critic, masks, joint_actions)
        return values, advantages

    def evaluate_centered_dae(self, cent_obs, rnn_states_critic, masks, joint_actions, joint_action_probs):
        """
        Evaluate exactly policy-centered DAE advantage.
        """
        values, advantages, _ = self.critic.evaluate_centered_dae(cent_obs,
                                                                  rnn_states_critic,
                                                                  masks,
                                                                  joint_actions,
                                                                  joint_action_probs)
        return values, advantages

    def evaluate_dae_sequence(self, cent_obs, rnn_states_critic, masks, joint_actions,
                              joint_action_probs=None, chunk_length=None):
        """
        Evaluate value and (optionally centered) DAE advantage over full time-major
        sequences, recomputing critic RNN states with truncated BPTT.
        """
        return self.critic.evaluate_dae_sequence(cent_obs,
                                                 rnn_states_critic,
                                                 masks,
                                                 joint_actions,
                                                 joint_action_probs,
                                                 chunk_length)

    def evaluate_factor_dae_sequence(self, cent_obs, rnn_states_critic, masks, own_index, own_actions,
                                     own_action_probs=None, prefix_onehot=None, prefix_mask=None,
                                     chunk_length=None):
        """
        Evaluate value and each agent's own (optionally ordered) centered DAE
        advantage factor over full time-major sequences.
        """
        return self.critic.evaluate_factor_dae_sequence(cent_obs,
                                                        rnn_states_critic,
                                                        masks,
                                                        own_index,
                                                        own_actions,
                                                        own_action_probs,
                                                        prefix_onehot,
                                                        prefix_mask,
                                                        chunk_length)

    def evaluate_gpae_sequence(self, cent_obs, rnn_states_critic, masks, own_index, own_actions,
                               own_action_probs, prefix_onehot, prefix_mask, chunk_length=None):
        """
        Evaluate E_Q^k(s, a_-k) and the last-position (full-prefix) centered
        factor for the GPAE auxiliary loss.
        """
        return self.critic.evaluate_gpae_sequence(cent_obs,
                                                  rnn_states_critic,
                                                  masks,
                                                  own_index,
                                                  own_actions,
                                                  own_action_probs,
                                                  prefix_onehot,
                                                  prefix_mask,
                                                  chunk_length)

    def get_action_probs(self, obs, rnn_states_actor, masks, available_actions=None):
        """
        Get factorized policy probabilities for exact DAE centering.
        """
        return self.actor.get_action_probs(obs, rnn_states_actor, masks, available_actions)

    def get_action_probs_sequence(self, obs, rnn_states_actor, masks, available_actions=None,
                                  chunk_length=None):
        """
        Recompute action probabilities over full time-major sequences with the
        current actor (for replayed data with stale stored hidden states).
        """
        return self.actor.get_action_probs_sequence(obs, rnn_states_actor, masks,
                                                    available_actions, chunk_length)

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param action: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs,
                                                                     rnn_states_actor,
                                                                     action,
                                                                     masks,
                                                                     available_actions,
                                                                     active_masks)

        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor
