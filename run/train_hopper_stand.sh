# python3 train.py task=hopper_stand exp_id=1 exp_group=hopper_stand

python3 scripts/generate_dmc_video.py --env hopper_stand --episode_len 60

python3 train_cmc.py task=hopper_stand exp_id=1 episode_len=60

python3 train_rl.py task=hopper_stand exp_id=1 episode_len=60


