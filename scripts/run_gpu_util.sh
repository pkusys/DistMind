FILENAME=tmp/GPU-stats.csv

nvidia-smi --query-gpu=timestamp,index,utilization.gpu --format=csv --loop-ms=100 --filename=$FILENAME