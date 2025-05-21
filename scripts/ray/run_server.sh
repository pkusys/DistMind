# !/bin/bash
source settings/config.sh

set -e  # Exit on error
python ./source/ray_benchmark/pt_example.py --n_gpus $WORLD_SIZE
