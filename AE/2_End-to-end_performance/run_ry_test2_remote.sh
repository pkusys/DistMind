#!/bin/bash
source settings/config.sh

# Create log directory
mkdir -p tmp/test2/ray
python ./source/py_utils/launch_remote.py --launch_part create_dir  --log_dir tmp/test2/ray --test_index 2
sleep 5

# Function to kill all subprocesses
cleanupserver() {
    echo "Cleaning up all background processes..."
    python ./source/py_utils/launch_remote.py --launch_part ray_server --test_index 2 --stop

    for pid in $PID_CONTROLLER; do
        if kill -0 $pid 2>/dev/null; then
            kill -9 -$pid
            sleep 2
        fi
    done
    wait
    echo "All processes finished and cleaned up."
}

cleanup() {
    echo "Cleaning up..."
    cleanupserver
}

# Trap SIGINT (Ctrl+C) and call cleanup function
trap cleanup SIGINT
trap cleanup EXIT

# Step1 : generate
echo "Running generate..."
./scripts/ray/run_generate.sh
sleep 1

# Define different experiment parameters
zipf_range=(0 0.9 0.99)
reschedule=(0.5 0.1)

# Function to run steps 4-9 with specific parameters
run_experiment() {
    local zipf=$1
    local resched=$2

    mkdir -p tmp/test2/ray/zipf${zipf}_resched${resched}

    echo "================================================================================="
    echo "Running experiment with parameter set: zipf=$zipf and resched=$resched"
    echo "================================================================================="

    # Step2 : run server
    echo "Running server..."
    python ./source/py_utils/launch_remote.py --launch_part ray_server --test_index 2 --model_seed res --n_models $GLOBAL_NUM_MODELS

    echo "Waiting for server to start..."
    sleep 60  # Give server a bit of head start

    # Step 3: Controller
    echo "Starting controller..."
    setsid python ./source/ray_benchmark/controller/controller.py --config settings/ray_controller.json --zipf_s $zipf --rescheduling_period $resched > tmp/test2/ray/zipf${zipf}_resched${resched}/log_controller.txt 2>&1 &
    PID_CONTROLLER=$!

    sleep 10

    # Step 4: client
    echo "Starting client..."
    python ./source/ray_benchmark/generate_requests.py --hostfile settings/serverhost_list.txt --output-stats tmp/test2/ray/zipf${zipf}_resched${resched}/stats.txt --output-req-log tmp/test2/ray/zipf${zipf}_resched${resched}/req_log.txt --controller-ip $GLOBAL_CONTROLLER_IP --controller-port $GLOBAL_CONTROLLER_PORT_FOR_SUBSCRIBER --max-requests $GLOBAL_NUM_REQUEST

    # Step 5: gather results
    echo "Gathering latency..."
    python ./source/ray_benchmark/check_latency.py --logdir tmp/test2/ray/zipf${zipf}_resched${resched} 2>&1 | tee tmp/test2/ray/zipf${zipf}_resched${resched}/check_latency.txt

    echo "Gathering stats..."
    python ./source/ray_benchmark/aggregate_stats.py --log tmp/test2/ray/zipf${zipf}_resched${resched}/stats.txt --output tmp/test2/ray/zipf${zipf}_resched${resched}/aggregated_stats.txt 2>&1 | tee tmp/test2/ray/zipf${zipf}_resched${resched}/avg_stats.txt

    cleanupserver
}

# Run experiments with different zipf and reschedule parameters
for i in "${!zipf_range[@]}"; do
    zipf=${zipf_range[$i]}
    resched=1
    
    run_experiment $zipf $resched
    
    # Wait a bit between runs to ensure clean state
    sleep 30
done

# Run experiments with different zipf and reschedule parameters
for i in "${!reschedule[@]}"; do
    zipf=0.9
    resched=${reschedule[$i]}
    
    run_experiment $zipf $resched
    
    # Wait a bit between runs to ensure clean state
    sleep 30
done
