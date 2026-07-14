#!/bin/sh
# DAE hyperparameter sweep: entropy coefficient and network capacity (no GAE->DAE warmup).
# Comment out variants you don't need; each run is 5M steps.
env="StarCraft2"
map="3s5z"
algo="rmappo"
seed_max=1

common="--env_name ${env} --algorithm_name ${algo} --map_name ${map} \
--n_training_threads 1 --n_rollout_threads 32 --num_mini_batch 1 --episode_length 100 \
--num_env_steps 5000000 --use_stacked_frames --ppo_epoch 5 --clip_param 0.2 \
--use_value_active_masks --use_eval --eval_episodes 32"
dae="--use_dae --dae_epoch 6 --dae_num_mini_batch 8 --dae_warmup_updates 0"

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"

    # --- entropy coefficient sweep (base network) ---
    for ent in 0.01 0.02 0.05; do
        CUDA_VISIBLE_DEVICES=0 python ../train/train_smac.py ${common} ${dae} \
            --entropy_coef ${ent} --experiment_name "dae_ent${ent}" --seed ${seed}
    done

    # --- capacity sweep (entropy 0.01; update after the entropy sweep if needed) ---
    # wider trunk (actor + critic)
    CUDA_VISIBLE_DEVICES=0 python ../train/train_smac.py ${common} ${dae} \
        --hidden_size 128 --experiment_name "dae_h128" --seed ${seed}
    # dedicated advantage-head branch only (isolates DAE-specific capacity)
    CUDA_VISIBLE_DEVICES=0 python ../train/train_smac.py ${common} ${dae} \
        --dae_head_hidden_size 256 --experiment_name "dae_head256" --seed ${seed}
    # both
    CUDA_VISIBLE_DEVICES=0 python ../train/train_smac.py ${common} ${dae} \
        --hidden_size 128 --dae_head_hidden_size 256 --experiment_name "dae_h128_head256" --seed ${seed}

    # --- GAE baselines at both widths (fair capacity comparison, as in the DAE paper) ---
    CUDA_VISIBLE_DEVICES=0 python ../train/train_smac.py ${common} \
        --experiment_name "gae_h64" --seed ${seed}
    CUDA_VISIBLE_DEVICES=0 python ../train/train_smac.py ${common} \
        --hidden_size 128 --experiment_name "gae_h128" --seed ${seed}
done
