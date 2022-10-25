# python3 scripts/generate_metaworld_video.py --env plate_slide --episode_len 50 --im-w 224 --im-h 224 --video-dir plate_slide_224 --num-train 1500 --num-valid 150

python3 train_cmc.py task=plate_slide_224 exp_id=1

python3 train_rlv2.py task=plate_slide_224
