python3 scripts/generate_metaworld_video.py --env button_press --episode_len 50 --im-w 84 --im-h 84 --video-dir button_press_84

python3 train_cmc.py task=button_press_84 exp_id=1

python3 train_rlv2.py task=button_press_84
