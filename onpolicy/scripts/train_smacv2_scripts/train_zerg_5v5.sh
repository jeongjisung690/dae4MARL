env="StarCraft2v2"
map="10gen_zerg"
algo="mat"
units="5v5"
wandb_project="StarCraft2v2"

exp="tune2"
seed_max=3

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python ../train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --map_name ${map} --seed ${seed} --units ${units} --n_training_threads 1 --n_rollout_threads 32 --num_mini_batch 1 --episode_length 100 \
    --num_env_steps 10000000 --use_stacked_frames --ppo_epoch 10 --clip_param 0.05 --use_value_active_masks --use_eval --eval_episodes 32 --wandb_project ${wandb_project}
done
