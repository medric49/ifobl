# python3 train.py task=reacher_easy exp_id=1 exp_group=reacher_easy

python3 scripts/generate_dmc_video.py --env reacher_easy --episode_len 60

python3 train_cmc.py task=reacher_easy exp_id=1 episode_len=60

python3 train_rl.py task=reacher_easy exp_id=1 episode_len=60
