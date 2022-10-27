python3 scripts/generate_metaworld_video.py --env drawer_close --episode_len 50 --im-w 224 --im-h 224 --video-dir drawer_close_224 --num-train 1500 --num-valid 150

python3 train_cmc.py task=drawer_close_224 exp_id=1

python3 train_rlv2.py task=drawer_close_224
