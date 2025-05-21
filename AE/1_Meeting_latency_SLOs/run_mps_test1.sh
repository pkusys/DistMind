#!/bin/bash
source settings/config.sh
export CUDA_MEMPOOL_ALLOW_INTERNAL=1
export CUDA_MANAGED_FORCE_DEVICE_ALLOC=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -e  # Exit on error
# Set default values for parameters
# $1: NUM_MODELS - default is GLOBAL_NUM_MODELS from config.sh
# $2: MODEL_SEED - default is 'res' (for resnet)
NUM_MODELS=${1:-$GLOBAL_NUM_MODELS}
MODEL_SEED=${2:-"res"}

# Create log directory
mkdir -p tmp/test1/mps/$MODEL_SEED

# Function to kill all subprocesses
cleanup() {
    echo "Cleaning up all background processes..."
    for pid in $PID_LB $PID_SERVER $PID_CONTROLLER; do
        if kill -0 $pid 2>/dev/null; then
            kill -9 -$pid
            sleep 2
        fi
    done
    wait
    echo "All processes finished and cleaned up."
    # Stop MPS
    echo "Stopping MPS..."
    ./scripts/mps/run_shutdown.sh
}

# Trap SIGINT (Ctrl+C) and call cleanup function
trap cleanup SIGINT
trap cleanup EXIT

# Step 1: Start MPS
echo "Starting MPS..."
./scripts/mps/run_mps.sh

sleep 10 # Give MPS a bit of time to start

# Step 2: Generate
echo "Running generate..."
./scripts/mps/run_generate.sh $NUM_MODELS $MODEL_SEED

# Step 3: Load Model
echo "Loading model..."
./scripts/mps/run_load_models.sh

# Step 4: Controller
echo "Starting controller..."
setsid python build/bin/controller.py --config ./settings/mps_controller.json > tmp/test1/mps/$MODEL_SEED/log_controller.txt 2>&1 &
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

setsid python source/mps/load_balancer.py --controller-address $CONTROLLER_IP --controller-port $CONTROLLER_PORT --client-port  $PORT_FOR_CLIENT --server-port  $PORT_FOR_SERVER > tmp/test1/mps/$MODEL_SEED/log_LB.txt 2>&1 &
PID_LB=$!

sleep 10  # Give load balancer a bit of head start

# Step 7: Server
echo "Starting server..."
IP_FOR_LOAD_BALANCER=$GLOBAL_LOAD_BALANCER_IP
LOCAL_IP=$GLOBAL_LOCAL_IP
CONTROLLER_PORT_SERVER=$GLOBAL_CONTROLLER_PORT_FOR_SERVER

echo "IP for client: $IP_FOR_LOAD_BALANCER"
echo "Port for servers: $PORT_FOR_SERVER"
echo "Local IP: $LOCAL_IP"
echo "Controller IP: $CONTROLLER_IP"
echo "Controller port: $CONTROLLER_PORT_SERVER"

setsid python source/mps/server_agent.py --lb-port $PORT_FOR_SERVER --lb-ip $IP_FOR_LOAD_BALANCER --size-list build/mps/model_sizes.txt --gpu-num $WORLD_SIZE --ctrl-ip $CONTROLLER_IP --ctrl-port $CONTROLLER_PORT_SERVER --local-ip $LOCAL_IP --log-dir ../../tmp/test1/mps/$MODEL_SEED/inference_logs > tmp/test1/mps/$MODEL_SEED/log_server.txt 2>&1 &
PID_SERVER=$!

# Step 8: Wait for readiness
echo "Waiting 120 seconds before running client..."
sleep 120

# Step 9: Run client
echo "Running client inference..."
./scripts/run_client_max_inference.sh > tmp/test1/mps/$MODEL_SEED/log_client.txt

