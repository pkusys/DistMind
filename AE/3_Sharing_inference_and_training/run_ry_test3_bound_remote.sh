#!/bin/bash
source settings/config.sh

set -e  # Exit on error

PRIFIX=${1-"tmp/test3/bound"}
# Create log directory
mkdir -p $PRIFIX/ray
python ./source/py_utils/launch_remote.py --launch_part create_dir  --log_dir $PRIFIX/ray --test_index 3
sleep 5

# Function to kill all subprocesses
cleanup() {
    echo "Cleaning up all background processes..."
    python ./source/py_utils/launch_remote.py --launch_part ray_server --test_index 3 --stop

    for pid in $PID_CONTROLLER; do
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


# Step1 : generate
echo "Running generate..."
./scripts/ray/run_generate.sh
sleep 1

# Step2 : run server
echo "Running server..."
python ./source/py_utils/launch_remote.py --launch_part ray_server --test_index 3 --model_seed res --n_models $GLOBAL_NUM_MODELS --log_file $PRIFIX/ray/log_server.txt

echo "Waiting for server to start..."
sleep 60  # Give server a bit of head start

# Step 3: Controller
echo "Starting controller..."
setsid python ./source/ray_benchmark/controller/controller.py --config settings/ray_controller.json --use_train True > $PRIFIX/ray/log_controller.txt 2>&1 &
PID_CONTROLLER=$!

sleep 10

# Step 4: client
echo "Starting client..."
setsid python ./source/ray_benchmark/generate_requests.py --hostfile settings/serverhost_list.txt --output-stats $PRIFIX/ray/stats.txt --output-req-log $PRIFIX/ray/req_log.txt --controller-ip $GLOBAL_CONTROLLER_IP --controller-port $GLOBAL_CONTROLLER_PORT_FOR_SUBSCRIBER --max-requests 0 > $PRIFIX/ray/log_client.txt 2>&1 &
PID_CLIENT=$!

sleep 400  # Give client a bit of time to finish
kill -9 -$PID_CLIENT

sleep 5

# Step 5: gather results
echo "Gathering stats..."
python ./source/ray_benchmark/aggregate_stats.py --log $PRIFIX/ray/stats.txt --output $PRIFIX/ray/aggregated_stats.txt 2>&1 | tee $PRIFIX/ray/avg_stats.txt
