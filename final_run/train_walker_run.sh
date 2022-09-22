TASK='walker_run'

# Train expert
python3 train.py task=$TASK exp_group=$TASK exp_id=1

# Generate expert videos
python3 scripts/generate_dmc_video.py --env $TASK --episode_len 60

# Pretrain trajectory encoder
python3 train_cmc.py task=$TASK exp_id=1

# Train agent
python3 train_rlv2.py task=$TASK
