import wandb
import os
import numpy as np
import torch
from tensorboardX import SummaryWriter
from onpolicy.utils.shared_buffer import SharedReplayBuffer

def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()

class _OverrideArgs(object):
    """Proxy that overrides selected attributes and delegates the rest to the
    wrapped args. Copying the args object itself is not safe: with wandb it is
    a wandb.Config, and copy.copy leaves it uninitialized (infinite recursion
    in its __getattr__)."""
    def __init__(self, base, **overrides):
        self._base = base
        for key, value in overrides.items():
            setattr(self, key, value)

    def __getattr__(self, name):
        return getattr(self.__dict__["_base"], name)

class Runner(object):
    """
    Base class for training recurrent policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']
        if config.__contains__("render_envs"):
            self.render_envs = config['render_envs']       

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
            self.run_dir = str(wandb.run.dir)
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            from onpolicy.algorithms.mat.mat_trainer import MATTrainer as TrainAlgo
            from onpolicy.algorithms.mat.algorithm.transformer_policy import TransformerPolicy as Policy
        else:
            from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
            from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy

        share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]

        print("obs_space: ", self.envs.observation_space)
        print("share_obs_space: ", self.envs.share_observation_space)
        print("act_space: ", self.envs.action_space)
        self.all_args.num_agents = self.num_agents
        
        # policy network
        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            self.policy = Policy(self.all_args, self.envs.observation_space[0], share_observation_space, self.envs.action_space[0], self.num_agents, device = self.device)
        else:
            self.policy = Policy(self.all_args, self.envs.observation_space[0], share_observation_space, self.envs.action_space[0], device = self.device)

        if self.model_dir is not None:
            self.restore(self.model_dir)

        # algorithm
        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            self.trainer = TrainAlgo(self.all_args, self.policy, self.num_agents, device = self.device)
        else:
            self.trainer = TrainAlgo(self.all_args, self.policy, device = self.device)
        
        # buffer
        self.buffer = SharedReplayBuffer(self.all_args,
                                        self.num_agents,
                                        self.envs.observation_space[0],
                                        share_observation_space,
                                        self.envs.action_space[0])

        # rollout accumulation: aggregate buffer holding rollout_accumulation
        # sequential chunks side by side along the thread dimension, so training
        # sees an effective batch of n_rollout_threads * rollout_accumulation envs.
        self.rollout_accumulation = getattr(self.all_args, "rollout_accumulation", 1)
        if self.rollout_accumulation > 1:
            agg_args = _OverrideArgs(self.all_args,
                                     n_rollout_threads=self.n_rollout_threads * self.rollout_accumulation)
            self.agg_buffer = SharedReplayBuffer(agg_args,
                                                 self.num_agents,
                                                 self.envs.observation_space[0],
                                                 share_observation_space,
                                                 self.envs.action_space[0])
            self.agg_buffer.next_value = np.zeros_like(self.agg_buffer.value_preds[-1])
            self._accumulated_chunks = 0

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        raise NotImplementedError

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError

    def insert(self, data):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        raise NotImplementedError
    
    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                        np.concatenate(self.buffer.obs[-1]),
                                                        np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                        np.concatenate(self.buffer.masks[-1]))
        else:
            next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                        np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                        np.concatenate(self.buffer.masks[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)
    
    @property
    def train_buffer(self):
        """Buffer the training update actually consumes."""
        if self.rollout_accumulation > 1:
            return self.agg_buffer
        return self.buffer

    def accumulate(self, chunk):
        """Copy the collection buffer (with returns already computed) into the
        aggregate buffer at the thread slice belonging to this chunk."""
        n = self.n_rollout_threads
        env_slice = slice(chunk * n, (chunk + 1) * n)
        agg, buf = self.agg_buffer, self.buffer
        agg.share_obs[:, env_slice] = buf.share_obs
        agg.obs[:, env_slice] = buf.obs
        agg.rnn_states[:, env_slice] = buf.rnn_states
        agg.rnn_states_critic[:, env_slice] = buf.rnn_states_critic
        agg.value_preds[:, env_slice] = buf.value_preds
        agg.returns[:, env_slice] = buf.returns
        agg.advantages[:, env_slice] = buf.advantages
        agg.actions[:, env_slice] = buf.actions
        agg.action_log_probs[:, env_slice] = buf.action_log_probs
        agg.rewards[:, env_slice] = buf.rewards
        agg.masks[:, env_slice] = buf.masks
        agg.bad_masks[:, env_slice] = buf.bad_masks
        agg.active_masks[:, env_slice] = buf.active_masks
        if buf.available_actions is not None:
            agg.available_actions[:, env_slice] = buf.available_actions
        if getattr(buf, "anomaly_masks", None) is not None:
            agg.anomaly_masks[:, env_slice] = buf.anomaly_masks
        agg.next_value[env_slice] = buf.next_value
        self._accumulated_chunks += 1

    def train(self):
        """Train policies with data in buffer. """
        if self.rollout_accumulation > 1:
            # guards against runners that run with rollout_accumulation > 1 but
            # never call accumulate(), which would train on an all-zero buffer
            assert self._accumulated_chunks == self.rollout_accumulation, (
                "rollout_accumulation={} but only {} chunks were accumulated before train(); "
                "this runner does not implement rollout accumulation.".format(
                    self.rollout_accumulation, self._accumulated_chunks))
            self._accumulated_chunks = 0
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.train_buffer)
        self.buffer.after_update()
        return train_infos

    def save(self, episode=0):
        """Save policy's actor and critic networks."""
        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            self.policy.save(self.save_dir, episode)
        else:
            policy_actor = self.trainer.policy.actor
            torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor.pt")
            policy_critic = self.trainer.policy.critic
            torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic.pt")

    def restore(self, model_dir):
        """Restore policy's networks from a saved model."""
        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            self.policy.restore(model_dir)
        else:
            policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor.pt')
            self.policy.actor.load_state_dict(policy_actor_state_dict)
            if not self.all_args.use_render:
                policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic.pt')
                self.policy.critic.load_state_dict(policy_critic_state_dict)

    def log_train(self, train_infos, total_num_steps):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v)>0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
