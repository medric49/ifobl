python3 scripts/generate_dmc_video.py --env finger_turn_easy --episode_len 60

python3 train_cmc.py task=finger_turn_easy exp_id=1 episode_len=60

python3 train_rl.py task=finger_turn_easy exp_id=1 episode_len=60
