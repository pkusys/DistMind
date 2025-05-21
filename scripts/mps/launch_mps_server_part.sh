#!/bin/bash
source settings/config.sh
export CUDA_MEMPOOL_ALLOW_INTERNAL=1
export CUDA_MANAGED_FORCE_DEVICE_ALLOC=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -e  # Exit on error
# Set default values for parameters
# $1: NUM_MODELS - default is GLOBAL_NUM_MODELS from config.sh
# $2: MODEL_SEED - default is 'res' (for resnet)
TEST_INDEX=${1:-1}
NUM_MODELS=${2:-$GLOBAL_NUM_MODELS}
MODEL_SEED=${3:-"res"}
TRAINING_LOG_DIR=${4:-"None"}
INFERENCE_LOG_DIR=${5:-"None"}

mkdir -p tmp/test$TEST_INDEX/mps/$MODEL_SEED

# Function to kill all subprocesses
cleanup() {
    # Cleanup all processes
    echo "Cleaning up all background processes..."
    for pid in $PID_SERVER ; do
        if kill -0 $pid 2>/dev/null; then
            kill -9 -$pid
            sleep 2
        fi
    done

    wait
    echo "All processes finished and cleaned up."
    rm -rf .pid_mps_server
}

# Trap SIGINT (Ctrl+C) and call cleanup function
trap cleanup SIGINT
trap cleanup EXIT

# Step 7: Server
echo "Starting server..."
IP_FOR_LOAD_BALANCER=$GLOBAL_LOAD_BALANCER_IP
LOCAL_IP=$GLOBAL_LOCAL_IP
CONTROLLER_IP=$GLOBAL_CONTROLLER_IP
PORT_FOR_SERVER=$GLOBAL_LOAD_BALANCER_PORT_FOR_SERVER
CONTROLLER_PORT_SERVER=$GLOBAL_CONTROLLER_PORT_FOR_SERVER

echo "IP for client: $IP_FOR_LOAD_BALANCER"
echo "Port for servers: $PORT_FOR_SERVER"
echo "Local IP: $LOCAL_IP"
echo "Controller IP: $CONTROLLER_IP"
echo "Controller port: $CONTROLLER_PORT_SERVER"

if [ "$TRAINING_LOG_DIR" != "None" ]; then
    setsid python source/mps/server_agent.py --lb-port $PORT_FOR_SERVER --lb-ip $IP_FOR_LOAD_BALANCER --size-list build/mps/model_sizes.txt --gpu-num $WORLD_SIZE --ctrl-ip $CONTROLLER_IP --ctrl-port $CONTROLLER_PORT_SERVER --local-ip $LOCAL_IP --training-log-dir $TRAINING_LOG_DIR > tmp/test$TEST_INDEX/mps/$MODEL_SEED/log_server.txt 2>&1 &
    PID_SERVER=$!
else 
    if [ "$INFERENCE_LOG_DIR" != "None" ]; then
        setsid python source/mps/server_agent.py --lb-port $PORT_FOR_SERVER --lb-ip $IP_FOR_LOAD_BALANCER --size-list build/mps/model_sizes.txt --gpu-num $WORLD_SIZE --ctrl-ip $CONTROLLER_IP --ctrl-port $CONTROLLER_PORT_SERVER --local-ip $LOCAL_IP --log-dir $INFERENCE_LOG_DIR > tmp/test$TEST_INDEX/mps/$MODEL_SEED/log_server.txt 2>&1 &
        PID_SERVER=$!
    else
        setsid python source/mps/server_agent.py --lb-port $PORT_FOR_SERVER --lb-ip $IP_FOR_LOAD_BALANCER --size-list build/mps/model_sizes.txt --gpu-num $WORLD_SIZE --ctrl-ip $CONTROLLER_IP --ctrl-port $CONTROLLER_PORT_SERVER --local-ip $LOCAL_IP > tmp/test$TEST_INDEX/mps/$MODEL_SEED/log_server.txt 2>&1 &
        PID_SERVER=$!
    fi
fi

echo $PID_SERVER > .pid_mps_server
