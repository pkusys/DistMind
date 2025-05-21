#!/bin/bash
source settings/config.sh

set -e  # Exit on error
# Set default values for parameters
# $1: NUM_MODELS - default is GLOBAL_NUM_MODELS from config.sh
# $2: MODEL_SEED - default is 'res' (for resnet)
NUM_MODELS=${1:-$GLOBAL_NUM_MODELS}
MODEL_SEED=${2:-"res"}

# Create log directory
mkdir -p tmp/test1/ray/$MODEL_SEED

# Function to kill all subprocesses
cleanup() {
    echo "Cleaning up all background processes..."
    ps aux | grep ray | grep -v grep | awk '{print $2}' | xargs -r kill -9
    for pid in $PID_SERVER $PID_CONTROLLER; do
        if kill -0 $pid 2>/dev/null; then
            kill -9 -$pid
            sleep 2
        fi
    done
    wait
    echo "All processes finished and cleaned up."
}

# Trap SIGINT (Ctrl+C) and call cleanup function
trap cleanup SIGINT
trap cleanup EXIT

ps aux | grep ray | grep -v grep | awk '{print $2}' | xargs -r kill -9

# Step1 : generate
echo "Running generate..."
./scripts/ray/run_generate.sh $NUM_MODELS $MODEL_SEED
sleep 1

# Step2 : run server
echo "Running server..."
setsid python ./source/ray_benchmark/pt_example.py --n_gpus $WORLD_SIZE --n_model_variants $NUM_MODELS --model_name $MODEL_SEED > tmp/test1/ray/$MODEL_SEED/log_server.txt 2>&1 &
PID_SERVER=$!

echo "Waiting for server to start..."
sleep 60  # Give server a bit of head start

# Step 3: Controller
echo "Starting controller..."
setsid python ./source/ray_benchmark/controller/controller.py --config settings/ray_controller.json > tmp/test1/ray/$MODEL_SEED/log_controller.txt 2>&1 &
PID_CONTROLLER=$!

sleep 10

# Step 4: client
echo "Starting client..."
python ./source/ray_benchmark/generate_requests.py --hostfile settings/serverhost_list.txt --output-stats tmp/test1/ray/$MODEL_SEED/stats.txt --output-req-log tmp/test1/ray/$MODEL_SEED/req_log.txt --controller-ip $GLOBAL_CONTROLLER_IP --controller-port $GLOBAL_CONTROLLER_PORT_FOR_SUBSCRIBER --max-requests $GLOBAL_NUM_REQUEST

# Step 5: gather results
echo "Gathering latency..."
python ./source/ray_benchmark/check_latency.py --logdir tmp/test1/ray/$MODEL_SEED 2>&1 | tee tmp/test1/ray/$MODEL_SEED/check_latency.txt

echo "Gathering stats..."
python ./source/ray_benchmark/aggregate_stats.py --log tmp/test1/ray/$MODEL_SEED/stats.txt --output tmp/test1/ray/$MODEL_SEED/aggregated_stats.txt 2>&1 | tee tmp/test1/ray/$MODEL_SEED/avg_stats.txt
