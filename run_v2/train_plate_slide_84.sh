python3 scripts/generate_metaworld_video.py --env plate_slide --episode_len 50 --im-w 84 --im-h 84 --video-dir plate_slide_84

python3 train_cmc.py task=plate_slide_84 exp_id=1

python3 train_rlv2.py task=plate_slide_84
