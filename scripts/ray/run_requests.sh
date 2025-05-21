# !/bin/bash
source settings/config.sh

set -e  # Exit on error

mkdir -p tmp/ray_benchmark

python ./source/ray_benchmark/generate_requests.py --hostfile settings/serverhost_list.txt --output-stats tmp/ray_benchmark/stats.txt --output-req-log tmp/ray_benchmark/req_log.txt --controller-ip $GLOBAL_CONTROLLER_IP --controller-port $GLOBAL_CONTROLLER_PORT_FOR_SUBSCRIBER --max-requests $GLOBAL_NUM_REQUEST