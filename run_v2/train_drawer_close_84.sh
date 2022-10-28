python3 scripts/generate_metaworld_video.py --env drawer_close --episode_len 50 --im-w 84 --im-h 84 --video-dir drawer_close_84

python3 train_cmc.py task=drawer_close_84 exp_id=1

python3 train_rlv2.py task=drawer_close_84
