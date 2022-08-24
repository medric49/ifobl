# python3 scripts/generate_dmc_video.py --env reacher_hard --episode_len 200

python3 train_cmc.py task=reacher_hard exp_id=1 episode_len=80

python3 train_rl.py task=reacher_hard exp_id=1 episode_len=80


