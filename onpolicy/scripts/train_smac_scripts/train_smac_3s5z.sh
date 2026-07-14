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
    CUDA_VISIBLE_DEVICES=0 python ../train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --map_name ${map} --seed ${seed} --n_training_threads 1 --n_rollout_threads 32 --num_mini_batch 1 --episode_length 100 \
    --num_env_steps 5000000 --use_stacked_frames --ppo_epoch 5 --clip_param 0.2 --use_value_active_masks --use_eval --eval_episodes 32 --use_dae --dae_epoch 6 --dae_num_mini_batch 8 --hidden_size 128 --dae_head_hidden_size 256
done
##MAPPO
# --ppo_epoch 5, --clip_param 0.2
##MAT
# --ppo_epoch 10, --clip_param 0.05
