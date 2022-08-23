#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=8  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32G       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=10:00:00
#SBATCH --output=%N-%j.out
#SBATCH --account=rrg-eugenium
#SBATCH --mail-user=medric49@gmail.com
#SBATCH --mail-type=ALL

module load python
source venv/bin/activate

python train.py task=reacher_hard exp_id=1 exp_group=reacher_hard