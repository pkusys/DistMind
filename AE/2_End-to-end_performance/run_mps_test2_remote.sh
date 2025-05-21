#!/bin/bash
source settings/config.sh
export CUDA_MEMPOOL_ALLOW_INTERNAL=1
export CUDA_MANAGED_FORCE_DEVICE_ALLOC=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -e  # Exit on error

# Create log directory
mkdir -p tmp/test2/mps
python ./source/py_utils/launch_remote.py --launch_part create_dir  --log_dir tmp/test2/mps --test_index 2
sleep 5

# Function to kill all subprocesses
cleanServer() {
    echo "Cleaning up all server background processes..."
    python ./source/py_utils/launch_remote.py --launch_part mps_server --test_index 2 --stop
    sleep 5

    for pid in $PID_LB $PID_CONTROLLER; do
        if kill -0 $pid 2>/dev/null; then
            kill -9 -$pid
            sleep 2
        fi
    done
}

cleanup() {
    # Cleanup all processes
    cleanServer

    echo "All processes finished and cleaned up."
    # Stop MPS
    echo "Stopping MPS..."
    python ./source/py_utils/launch_remote.py --launch_part mps_storage --test_index 2 --stop
    sleep 5
}

# Trap SIGINT (Ctrl+C) and call cleanup function
trap cleanup SIGINT
trap cleanup EXIT

# Step 1: Prepare MPS
python ./source/py_utils/launch_remote.py --launch_part mps_storage --test_index 2 --model_seed res --n_models $GLOBAL_NUM_MODELS

sleep 10 # Give MPS a bit of time to start

# Step 2: Generate
echo "Running generate..."
./scripts/mps/run_generate.sh

# Define different experiment parameters
zipf_range=(0 0.9 0.99)
reschedule=(0.5 0.1)

# Function to run steps 4-9 with specific parameters
run_experiment() {
    local zipf=$1
    local resched=$2
    
    echo "================================================================================="
    echo "Running experiment with parameter set: zipf=$zipf and resched=$resched"
    echo "================================================================================="
    
    # Step 4: Controller
    echo "Starting controller..."
    setsid python build/bin/controller.py --config ./settings/mps_controller.json --use_train False --zipf_s $zipf --rescheduling_period $resched > tmp/test2/mps/log_controller_zipf${zipf}_resched${resched}.txt 2>&1 &
    PID_CONTROLLER=$!

    sleep 10  # Give controller a bit of head start

    # Step 6: Load Balancer
    echo "Starting load balancer..."
    PORT_FOR_CLIENT=$GLOBAL_LOAD_BALANCER_PORT_FOR_CLIENT
    PORT_FOR_SERVER=$GLOBAL_LOAD_BALANCER_PORT_FOR_SERVER
    CONTROLLER_IP=$GLOBAL_CONTROLLER_IP
    CONTROLLER_PORT=$GLOBAL_CONTROLLER_PORT_FOR_SUBSCRIBER

    echo "Controller IP: $CONTROLLER_IP"
    echo "Controller port: $CONTROLLER_PORT"
    echo "Port for client: $PORT_FOR_CLIENT"
    echo "Port for servers: $PORT_FOR_SERVER"

    setsid python source/mps/load_balancer.py --controller-address $CONTROLLER_IP --controller-port $CONTROLLER_PORT --client-port $PORT_FOR_CLIENT --server-port $PORT_FOR_SERVER > tmp/test2/mps/log_LB_zipf${zipf}_resched${resched}.txt 2>&1 &
    PID_LB=$!

    sleep 10  # Give load balancer a bit of head start

    # Step 7: Server
    echo "Starting server..."
    python ./source/py_utils/launch_remote.py --launch_part mps_server --test_index 2 --model_seed res --n_models $GLOBAL_NUM_MODELS

    # Step 8: Wait for readiness
    echo "Waiting 120 seconds before running client..."
    sleep 120

    # Step 9: Run client
    echo "Running client inference..."
    ./scripts/run_client_max_inference.sh > tmp/test2/mps/log_client_zipf${zipf}_resched${resched}.txt
    
    # Clean up servers after each run
    cleanServer
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

