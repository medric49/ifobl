python3 scripts/generate_dmc_video.py --env finger_turn_easy --episode_len 50 --im-w 84 --im-h 84 --video-dir finger_turn_easy_84

python3 train_cmc.py task=finger_turn_easy_84 exp_id=1 episode_len=50

python3 train_rl.py task=finger_turn_easy_84 exp_id=1 episode_len=50
