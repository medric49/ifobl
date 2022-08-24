python3 train.py task=cartpole_swingup exp_id=1 exp_group=cartpole_swingup

python3 scripts/generate_dmc_video.py --env cartpole_swingup --episode_len 60

python3 train_cmc.py task=cartpole_swingup exp_id=1 episode_len=60

python3 train_rl.py task=cartpole_swingup exp_id=1 episode_len=60


