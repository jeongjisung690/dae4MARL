#!/bin/sh
env="StarCraft2"
map="3s5z"
algo="rmappo"
exp="check"
seed_max=1

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    # rollout_accumulation 32: collect 32 sequential chunks with the 32 envs before each
    # update -> effective on-policy batch of 32*32=1024 rollout threads (matches the
    # official DAE setting) while only 32 SC2 processes exist.
    # num_mini_batch / dae_num_mini_batch are scaled by the same factor so the
    # per-gradient-step batch (and GPU memory) stays the same as the old
    # "32 threads, num_mini_batch 1, dae_num_mini_batch 8" setting.
    CUDA_VISIBLE_DEVICES=0 python ../train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --map_name ${map} --seed ${seed} --n_training_threads 1 --n_rollout_threads 32 --rollout_accumulation 1 --num_mini_batch 1 --episode_length 100 \
    --num_env_steps 5000000 --use_stacked_frames --ppo_epoch 5 --clip_param 0.2 --use_value_active_masks --use_eval --eval_episodes 32 --use_dae --dae_epoch 6 --dae_num_mini_batch 8 --dae_head ordered  --dae_actor_adv agent --dae_order fixed \
    --anomaly_agent_id -1 --dae_gpae_coef 0.5 --dae_replay_size 3 --dae_replay_updates 4 --dae_trace_eta 1.05
done
## Anomaly-injection diagnostic (GPAE paper Sec. 3.1): agent 0 is forced to take
## "stop" (action 1) with 5% probability; wandb logs 'anomaly_adv_gap'
## (= mean advantage of teammates minus the misbehaving agent at injected steps;
## higher = better per-agent credit assignment; joint advantage gives ~0) and
## 'anomaly_count' (injected steps per batch). Defaults: --anomaly_prob 0.05
## --anomaly_action 1. Remove --anomaly_agent_id (or set -1) for clean runs.
## DAE variants (append to the command above):
# J-DAE / F-DAE-global (current default): a
# --dae_head additive --dae_actor_adv joint

# F-DAE-agent  (per-agent factors):       
# --dae_head additive --dae_actor_adv agent

# O-DAE-global (function-class ablation):
# --dae_head ordered  --dae_actor_adv joint --dae_order fixed

# O-DAE-fixed:                           
# --dae_head ordered  --dae_actor_adv agent --dae_order fixed

# O-DAE-perm   (Shapley averaging):
# --dae_head ordered  --dae_actor_adv agent --dae_order permute --dae_perm_eval_samples 8

# O-DAE + GPAE aux (reward-grounded last-position factors, Kim et al. AAMAS 2026):
# --dae_head ordered  --dae_actor_adv agent --dae_order fixed --dae_gpae_coef 0.5
# (logs gpae_eq_loss / gpae_aux_loss / gpae_adv_rms / gpae_factor_corr; sweep coef 0.1-1.0)
# + off-policy head replay (V-trace residual recursion + DT-ISR GPAE traces;
#   actor stays on-policy PPO; keeps early high-entropy data in the head training):
# --dae_replay_size 4 --dae_replay_updates 4 --dae_trace_eta 1.05
# (logs dae_replay_loss / dae_replay_c_joint / dae_replay_rho_own; if dae_replay_c_joint
#  collapses toward 0 the snapshots are too stale -> reduce dae_replay_size)


##MAPPO
# --ppo_epoch 5, --clip_param 0.2
##MAT
# --ppo_epoch 10, --clip_param 0.05
