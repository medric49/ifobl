python3 scripts/generate_metaworld_video.py --env button_press_topdown --episode_len 50 --im-w 84 --im-h 84 --video-dir button_press_topdown_84

python3 train_cmc.py task=button_press_topdown_84 exp_id=1 episode_len=50

python3 train_rl.py task=button_press_topdown_84 exp_id=1 episode_len=50 num_encoder_train_frames=600000
