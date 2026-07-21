import argparse


def get_config():
    """
    The configuration parser for common hyperparameters of all environment. 
    Please reach each `scripts/train/<env>_runner.py` file to find private hyperparameters
    only used in <env>.

    Prepare parameters:
        --algorithm_name <algorithm_name>
            specifiy the algorithm, including `["rmappo", "mappo", "rmappg", "mappg", "trpo"]`
        --experiment_name <str>
            an identifier to distinguish different experiment.
        --seed <int>
            set seed for numpy and torch 
        --cuda
            by default True, will use GPU to train; or else will use CPU; 
        --cuda_deterministic
            by default, make sure random seed effective. if set, bypass such function.
        --n_training_threads <int>
            number of training threads working in parallel. by default 1
        --n_rollout_threads <int>
            number of parallel envs for training rollout. by default 32
        --n_eval_rollout_threads <int>
            number of parallel envs for evaluating rollout. by default 1
        --n_render_rollout_threads <int>
            number of parallel envs for rendering, could only be set as 1 for some environments.
        --num_env_steps <int>
            number of env steps to train (default: 10e6)
        --user_name <str>
            [for wandb usage], to specify user's name for simply collecting training data.
        --wandb_project <str>
            [for wandb usage], to specify the wandb project name. Defaults to env_name.
        --use_wandb
            [for wandb usage], by default True, will log date to wandb server. or else will use tensorboard to log data.
    
    Env parameters:
        --env_name <str>
            specify the name of environment
        --use_obs_instead_of_state
            [only for some env] by default False, will use global state; or else will use concatenated local obs.
    
    Replay Buffer parameters:
        --episode_length <int>
            the max length of episode in the buffer. 
    
    Network parameters:
        --share_policy
            by default True, all agents will share the same network; set to make training agents use different policies. 
        --use_centralized_V
            by default True, use centralized training mode; or else will decentralized training mode.
        --stacked_frames <int>
            Number of input frames which should be stack together.
        --hidden_size <int>
            Dimension of hidden layers for actor/critic networks
        --layer_N <int>
            Number of layers for actor/critic networks
        --use_ReLU
            by default True, will use ReLU. or else will use Tanh.
        --use_popart
            by default True, use PopArt to normalize rewards. 
        --use_valuenorm
            by default True, use running mean and std to normalize rewards. 
        --use_feature_normalization
            by default True, apply layernorm to normalize inputs. 
        --use_orthogonal
            by default True, use Orthogonal initialization for weights and 0 initialization for biases. or else, will use xavier uniform inilialization.
        --gain
            by default 0.01, use the gain # of last action layer
        --use_naive_recurrent_policy
            by default False, use the whole trajectory to calculate hidden states.
        --use_recurrent_policy
            by default, use Recurrent Policy. If set, do not use.
        --recurrent_N <int>
            The number of recurrent layers ( default 1).
        --data_chunk_length <int>
            Time length of chunks used to train a recurrent_policy, default 10.
    
    Optimizer parameters:
        --lr <float>
            learning rate parameter,  (default: 5e-4, fixed).
        --critic_lr <float>
            learning rate of critic  (default: 5e-4, fixed)
        --opti_eps <float>
            RMSprop optimizer epsilon (default: 1e-5)
        --weight_decay <float>
            coefficience of weight decay (default: 0)
    
    PPO parameters:
        --ppo_epoch <int>
            number of ppo epochs (default: 15)
        --use_clipped_value_loss 
            by default, clip loss value. If set, do not clip loss value.
        --clip_param <float>
            ppo clip parameter (default: 0.2)
        --num_mini_batch <int>
            number of batches for ppo (default: 1)
        --entropy_coef <float>
            entropy term coefficient (default: 0.01)
        --use_max_grad_norm 
            by default, use max norm of gradients. If set, do not use.
        --max_grad_norm <float>
            max norm of gradients (default: 0.5)
        --use_gae
            by default, use generalized advantage estimation. If set, do not use gae.
        --gamma <float>
            discount factor for rewards (default: 0.99)
        --gae_lambda <float>
            gae lambda parameter (default: 0.95)
        --use_proper_time_limits
            by default, the return value does consider limits of time. If set, compute returns with considering time limits factor.
        --use_huber_loss
            by default, use huber loss. If set, do not use huber loss.
        --use_value_active_masks
            by default True, whether to mask useless data in value loss.  
        --huber_delta <float>
            coefficient of huber loss.  
    
    PPG parameters:
        --aux_epoch <int>
            number of auxiliary epochs. (default: 4)
        --clone_coef <float>
            clone term coefficient (default: 0.01)
    
    Run parameters：
        --use_linear_lr_decay
            by default, do not apply linear decay to learning rate. If set, use a linear schedule on the learning rate
    
    Save & Log parameters:
        --save_interval <int>
            time duration between contiunous twice models saving.
        --log_interval <int>
            time duration between contiunous twice log printing.
    
    Eval parameters:
        --use_eval
            by default, do not start evaluation. If set`, start evaluation alongside with training.
        --eval_interval <int>
            time duration between contiunous twice evaluation progress.
        --eval_episodes <int>
            number of episodes of a single evaluation.
    
    Render parameters:
        --save_gifs
            by default, do not save render video. If set, save video.
        --use_render
            by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.
        --render_episodes <int>
            the number of episodes to render a given env
        --ifi <float>
            the play interval of each rendered image in saved video.
    
    Pretrained parameters:
        --model_dir <str>
            by default None. set the path to pretrained model.
    """
    parser = argparse.ArgumentParser(
        description='onpolicy', formatter_class=argparse.RawDescriptionHelpFormatter)

    # prepare parameters
    parser.add_argument("--algorithm_name", type=str,
                        default='mappo', choices=["rmappo", "mappo", "happo", "hatrpo", "mat", "mat_dec"])

    parser.add_argument("--experiment_name", type=str, default="check", help="an identifier to distinguish different experiment.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for numpy/torch")
    parser.add_argument("--cuda", action='store_false', default=True, help="by default True, will use GPU to train; or else will use CPU;")
    parser.add_argument("--cuda_deterministic",
                        action='store_false', default=True, help="by default, make sure random seed effective. if set, bypass such function.")
    parser.add_argument("--n_training_threads", type=int,
                        default=1, help="Number of torch threads for training")
    parser.add_argument("--n_rollout_threads", type=int, default=32,
                        help="Number of parallel envs for training rollouts")
    parser.add_argument("--rollout_accumulation", type=int, default=1,
                        help="Number of sequential rollout chunks collected (without any policy update in "
                             "between) before each training update. The chunks are concatenated along the "
                             "thread dimension, so the update batch is statistically equivalent to "
                             "n_rollout_threads * rollout_accumulation parallel envs while only "
                             "n_rollout_threads env processes exist. Currently implemented in the "
                             "shared-policy SMAC runner only.")
    parser.add_argument("--n_eval_rollout_threads", type=int, default=1,
                        help="Number of parallel envs for evaluating rollouts")
    parser.add_argument("--n_render_rollout_threads", type=int, default=1,
                        help="Number of parallel envs for rendering rollouts")
    parser.add_argument("--num_env_steps", type=int, default=10e6,
                        help='Number of environment steps to train (default: 10e6)')
    parser.add_argument("--user_name", type=str, default='j-jiseong-okayama-university', help="[for wandb usage], to specify user's name for simply collecting training data.")
    parser.add_argument("--wandb_project", type=str, default=None, help="[for wandb usage], to specify the wandb project name. Defaults to env_name.")
    parser.add_argument("--use_wandb", action='store_false', default=True, help="[for wandb usage], by default True, will log date to wandb server. or else will use tensorboard to log data.")

    # env parameters
    parser.add_argument("--env_name", type=str, default='StarCraft2', help="specify the name of environment")
    parser.add_argument("--use_obs_instead_of_state", action='store_true',
                        default=False, help="Whether to use global state or concatenated obs")

    # replay buffer parameters
    parser.add_argument("--episode_length", type=int,
                        default=200, help="Max length for any episode")

    # network parameters
    parser.add_argument("--share_policy", action='store_false',
                        default=True, help='Whether agent share the same policy')
    parser.add_argument("--use_centralized_V", action='store_false',
                        default=True, help="Whether to use centralized V function")
    parser.add_argument("--stacked_frames", type=int, default=1,
                        help="Dimension of hidden layers for actor/critic networks")
    parser.add_argument("--use_stacked_frames", action='store_true',
                        default=False, help="Whether to use stacked_frames")
    parser.add_argument("--hidden_size", type=int, default=64,
                        help="Dimension of hidden layers for actor/critic networks") 
    parser.add_argument("--layer_N", type=int, default=1,
                        help="Number of layers for actor/critic networks")
    parser.add_argument("--use_ReLU", action='store_false',
                        default=True, help="Whether to use ReLU")
    parser.add_argument("--use_popart", action='store_true', default=False, help="by default False, use PopArt to normalize rewards.")
    parser.add_argument("--use_valuenorm", action='store_false', default=True, help="by default True, use running mean and std to normalize rewards.")
    parser.add_argument("--use_feature_normalization", action='store_false',
                        default=True, help="Whether to apply layernorm to the inputs")
    parser.add_argument("--use_orthogonal", action='store_false', default=True,
                        help="Whether to use Orthogonal initialization for weights and 0 initialization for biases")
    parser.add_argument("--gain", type=float, default=0.01,
                        help="The gain # of last action layer")

    # recurrent parameters
    parser.add_argument("--use_naive_recurrent_policy", action='store_true',
                        default=False, help='Whether to use a naive recurrent policy')
    parser.add_argument("--use_recurrent_policy", action='store_false',
                        default=True, help='use a recurrent policy')
    parser.add_argument("--recurrent_N", type=int, default=1, help="The number of recurrent layers.")
    parser.add_argument("--data_chunk_length", type=int, default=10,
                        help="Time length of chunks used to train a recurrent_policy")

    # optimizer parameters
    parser.add_argument("--lr", type=float, default=5e-4,
                        help='learning rate (default: 5e-4)')
    parser.add_argument("--critic_lr", type=float, default=5e-4,
                        help='critic learning rate (default: 5e-4)')
    parser.add_argument("--opti_eps", type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument("--weight_decay", type=float, default=0)

    # trpo parameters
    parser.add_argument("--kl_threshold", type=float, 
                        default=0.01, help='the threshold of kl-divergence (default: 0.01)')
    parser.add_argument("--ls_step", type=int, 
                        default=10, help='number of line search (default: 10)')
    parser.add_argument("--accept_ratio", type=float, 
                        default=0.5, help='accept ratio of loss improve (default: 0.5)')

    # ppo parameters
    parser.add_argument("--ppo_epoch", type=int, default=15,
                        help='number of ppo epochs (default: 15)')
    parser.add_argument("--use_clipped_value_loss",
                        action='store_false', default=True, help="by default, clip loss value. If set, do not clip loss value.")
    parser.add_argument("--clip_param", type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument("--num_mini_batch", type=int, default=1,
                        help='number of batches for ppo (default: 1)')
    parser.add_argument("--entropy_coef", type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument("--value_loss_coef", type=float,
                        default=1, help='value loss coefficient (default: 0.5)')
    parser.add_argument("--use_max_grad_norm",
                        action='store_false', default=True, help="by default, use max norm of gradients. If set, do not use.")
    parser.add_argument("--max_grad_norm", type=float, default=10.0,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument("--use_gae", action='store_false',
                        default=True, help='use generalized advantage estimation')
    parser.add_argument("--gamma", type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help='gae lambda parameter (default: 0.95)')
    parser.add_argument("--use_proper_time_limits", action='store_true',
                        default=False, help='compute returns taking into account time limits')
    parser.add_argument("--use_huber_loss", action='store_false', default=True, help="by default, use huber loss. If set, do not use huber loss.")
    parser.add_argument("--use_value_active_masks",
                        action='store_false', default=True, help="by default True, whether to mask useless data in value loss.")
    parser.add_argument("--use_policy_active_masks",
                        action='store_false', default=True, help="by default True, whether to mask useless data in policy loss.")
    parser.add_argument("--huber_delta", type=float, default=10.0, help=" coefficience of huber loss.")
    parser.add_argument("--use_dae", action='store_true',
                        default=False, help="use Direct Advantage Estimation for MAPPO advantages")
    parser.add_argument("--dae_epoch", type=int, default=1,
                        help="number of critic/advantage-head DAE updates per training iteration")
    parser.add_argument("--dae_num_mini_batch", type=int, default=1,
                        help="number of rollout-thread minibatches for each DAE update")
    parser.add_argument("--dae_loss_coef", type=float, default=1.0,
                        help="coefficient for the DAE residual loss")
    parser.add_argument("--dae_normalize_advantages", action='store_false',
                        default=True, help="by default True, RMS-normalize DAE advantages before the PPO actor update. If set, do not normalize.")
    parser.add_argument("--dae_centering", type=str, default="exact",
                        choices=["none", "exact"], help="policy-centering method for DAE advantages")
    parser.add_argument("--dae_head_gain", type=float, default=1.0,
                        help="orthogonal-init gain of the DAE advantage head output layer. 1.0 matches the "
                             "well-performing 7/14 run (rtsr13vx); 0.1 is the official-DAE-style small init "
                             "that was silently introduced right after that run and degraded learning.")
    parser.add_argument("--dae_head_hidden_size", type=int, default=0,
                        help="hidden width of a dedicated MLP branch for the DAE advantage head "
                             "(critic features -> hidden -> N*|A| table). 0 keeps the single linear head.")
    parser.add_argument("--dae_warmup_updates", type=int, default=0,
                        help="number of training updates over which the actor advantage is linearly blended "
                             "from GAE (weight 1 at update 0) to DAE (weight 1 afterwards). 0 disables the warmup. "
                             "The DAE head is trained from the start regardless.")
    parser.add_argument("--dae_head", type=str, default="additive",
                        choices=["additive", "ordered"],
                        help="advantage head function class. 'additive': per-agent factors A_i(s, a_i). "
                             "'ordered': factors conditioned on preceding agents' actions A_i(s, a_<i, a_i) "
                             "(Multi-Agent Advantage Decomposition / O-DAE).")
    parser.add_argument("--dae_actor_adv", type=str, default="joint",
                        choices=["joint", "agent"],
                        help="advantage signal passed to the actor. 'joint': the summed joint advantage, "
                             "identical for all agents. 'agent': each agent's own centered factor.")
    parser.add_argument("--dae_order", type=str, default="fixed",
                        choices=["fixed", "permute"],
                        help="agent ordering for the ordered DAE head. 'fixed': agent-index order. "
                             "'permute': a fresh random permutation per rollout thread on every evaluation.")
    parser.add_argument("--dae_adv_norm_scope", type=str, default="global",
                        choices=["global", "agent"],
                        help="scope of the RMS normalization of DAE actor advantages. 'global' preserves "
                             "relative credit scale between agents; 'agent' normalizes each agent separately.")
    parser.add_argument("--dae_perm_eval_samples", type=int, default=1,
                        help="with dae_order=permute, number of permutations averaged when computing actor "
                             "advantages (Shapley-style permutation averaging). Training still uses one "
                             "permutation per rollout thread per update.")
    parser.add_argument("--dae_gpae_coef", type=float, default=0.0,
                        help="weight of the GPAE auxiliary loss that ties the ordered head's "
                             "last-position (fully conditioned) factor A_k(s, a_-k, a_k) to a "
                             "reward-grounded per-agent GPAE target. The target comes from a "
                             "dedicated counterfactual-value head E_Q^k(s, a_-k) trained with the "
                             "per-agent n-step recursion of Kim et al. (GPAE, AAMAS 2026). "
                             "0 (default) disables the head and the loss. Requires dae_head=ordered "
                             "and dae_centering=exact.")
    parser.add_argument("--dae_gpae_lambda", type=float, default=0.95,
                        help="lambda of the per-agent GPAE advantage recursion "
                             "(bias-variance trade-off, as in GAE).")
    parser.add_argument("--dae_replay_size", type=int, default=0,
                        help="number of past rollout snapshots kept for off-policy training of the "
                             "DAE/GPAE heads (the actor stays on-policy PPO). 0 (default) disables replay. "
                             "Replayed value targets use a V-trace-style truncated joint importance ratio "
                             "on the difference-reward recursion; GPAE targets use DT-ISR traces.")
    parser.add_argument("--dae_replay_updates", type=int, default=0,
                        help="head updates per training iteration drawn from the replay store "
                             "(each pass consumes one snapshot, split into dae_num_mini_batch minibatches).")
    parser.add_argument("--dae_trace_eta", type=float, default=1.05,
                        help="eta of the double-truncated importance sampling ratio (DT-ISR, GPAE paper "
                             "Eq. 8): cap on the OTHER agents' joint ratio inside the per-agent trace "
                             "c^k = lambda * min(1, rho^k * min(eta, rho^-k)).")

    # anomaly-injection diagnostic (GPAE paper Sec. 3.1): force one agent to take a fixed
    # suboptimal action with some probability and log the advantage gap
    # dA = mean_{j != i} A_j - A_i at the injected steps. A larger gap means the
    # per-agent advantage correctly penalizes the misbehaving agent.
    parser.add_argument("--anomaly_agent_id", type=int, default=-1,
                        help="agent index whose actions are randomly overridden for the credit-assignment "
                             "diagnostic. -1 (default) disables the diagnostic.")
    parser.add_argument("--anomaly_prob", type=float, default=0.05,
                        help="per-step probability of overriding the anomaly agent's action.")
    parser.add_argument("--anomaly_action", type=int, default=1,
                        help="discrete action injected at anomaly steps (SMAC: 1 = stop).")

    # run parameters
    parser.add_argument("--use_linear_lr_decay", action='store_true',
                        default=False, help='use a linear schedule on the learning rate')
    # save parameters
    parser.add_argument("--save_interval", type=int, default=1, help="time duration between contiunous twice models saving.")

    # log parameters
    parser.add_argument("--log_interval", type=int, default=5, help="time duration between contiunous twice log printing.")

    # eval parameters
    parser.add_argument("--use_eval", action='store_true', default=False, help="by default, do not start evaluation. If set`, start evaluation alongside with training.")
    parser.add_argument("--eval_interval", type=int, default=25, help="time duration between contiunous twice evaluation progress.")
    parser.add_argument("--eval_episodes", type=int, default=32, help="number of episodes of a single evaluation.")

    # render parameters
    parser.add_argument("--save_gifs", action='store_true', default=False, help="by default, do not save render video. If set, save video.")
    parser.add_argument("--use_render", action='store_true', default=False, help="by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.")
    parser.add_argument("--render_episodes", type=int, default=5, help="the number of episodes to render a given env")
    parser.add_argument("--ifi", type=float, default=0.1, help="the play interval of each rendered image in saved video.")

    # pretrained parameters
    parser.add_argument("--model_dir", type=str, default=None, help="by default None. set the path to pretrained model.")
    
    # add for transformer
    parser.add_argument("--encode_state", action='store_true', default=False)
    parser.add_argument("--n_block", type=int, default=1)
    parser.add_argument("--n_embd", type=int, default=64)
    parser.add_argument("--n_head", type=int, default=1)
    parser.add_argument("--dec_actor", action='store_true', default=False)
    parser.add_argument("--share_actor", action='store_true', default=False)

    # add for online multi-task
    parser.add_argument("--train_maps", type=str, nargs='+', default=None)
    parser.add_argument("--eval_maps", type=str, nargs='+', default=None)
    
    return parser
