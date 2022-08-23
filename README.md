# ifobl
Imitation from observation using behavioral learning


## Installation

```shell
conda env create -f env.yml
conda activate ifobl
```


## Training

* Train the expert with DrQv2
```shell
python3 train.py task=reacher_hard exp_id=1 exp_group=1
```

* Train generate video of the expert
```shell
python3 script/generate_dmc_video.py --env reacher_hard --episode_len 60
```

* Train sequence encoder
```shell
python3 train_cmc.py task=reacher_hard exp_id=1 episode_len=60
```

* Train agent
```shell
python3 train_rl.py task=reacher_hard exp_id=1 episode_len=60
```