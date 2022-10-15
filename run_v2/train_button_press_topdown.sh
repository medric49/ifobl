# python3 scripts/generate_metaworld_video.py --env button_press_topdown --episode_len 60

python3 train_cmc.py task=button_press_topdown exp_id=1

python3 train_rlv2.py task=button_press_topdown
