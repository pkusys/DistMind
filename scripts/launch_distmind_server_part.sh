#!/bin/bash
source settings/config.sh

set -e  # Exit on error

# Set default values for parameters
# $1: NUM_MODELS - default is GLOBAL_NUM_MODELS from config.sh
# $2: MODEL_SEED - default is 'res' (for resnet)
TEST_INDEX=${1:-1}
NUM_MODELS=${2:-$GLOBAL_NUM_MODELS}
MODEL_SEED=${3:-"res"}
SYSTEM_TYPE=${4:-"distmind_remote"}
CACHE_SIZE=${5:-4096000000}
CACHE_BLOCK_SIZE=${6:-4096000}

mkdir -p tmp/test$TEST_INDEX/$SYSTEM_TYPE
mkdir -p tmp/test$TEST_INDEX/$SYSTEM_TYPE/$MODEL_SEED

# Function to kill all subprocesses
cleanup() {
    # Cleanup all processes
    echo "Cleaning up all background processes..."
    for pid in ${PID_SERVER[@]}; do
        if kill -0 $pid 2>/dev/null; then
            kill -9 $pid
            sleep 2
        fi
    done

    for pid in $PID_CACHE ; do
        if kill -0 $pid 2>/dev/null; then
            kill -9 $pid
            sleep 2
        fi
    done

    wait
    echo "All processes finished and cleaned up."
    rm -rf .pid_cache
    rm -rf .pid_server
    # Step 11: Stop storage
    find /dev/shm/* -writable -exec rm -rf {} + 2>/dev/null
}

# Trap SIGINT (Ctrl+C) and call cleanup function
trap cleanup SIGINT
trap cleanup EXIT

# Step 7: Cache
echo "Starting cache..."
LOCAL_IP=$GLOBAL_LOCAL_IP
LOAD_BALANCER_IP=$GLOBAL_LOAD_BALANCER_IP
METADATA_IP=$GLOBAL_METADATA_IP
METADATA_PORT=$GLOBAL_METADATA_PORT

CACHE_PORT=$GLOBAL_CAHCE_PORT_FOR_SERVER
LOAD_BALANCER_PORT_CACHE=$GLOBAL_LOAD_BALANCER_PORT_FOR_CACHE

echo "Local IP: $LOCAL_IP"
echo "Cache port: $CACHE_PORT"
echo "Metadata IP: $METADATA_IP"
echo "Metadata port: $METADATA_PORT"
echo "Load balancer IP: $LOAD_BALANCER_IP"
echo "Load balancer port: $LOAD_BALANCER_PORT_CACHE"
echo "Cache size: $CACHE_SIZE"
echo "Cache block size: $CACHE_BLOCK_SIZE"

sleep 1
FI_EFA_ENABLE_SHM_TRANSFER=0 ./build/bin/cache $LOCAL_IP $CACHE_PORT $METADATA_IP $METADATA_PORT $LOAD_BALANCER_IP $LOAD_BALANCER_PORT_CACHE cachecache $CACHE_SIZE $CACHE_BLOCK_SIZE > tmp/test$TEST_INDEX/$SYSTEM_TYPE/$MODEL_SEED/log_cache.txt 2>&1 &
PID_CACHE=$!
echo $PID_CACHE > .pid_cache

sleep 10  # Give cache a bit of head start

# Step 8: Server
echo "Starting server..."

LOAD_BALANCER_PORT_SERVER=$GLOBAL_LOAD_BALANCER_PORT_FOR_SERVER
CONTROLLER_IP=$GLOBAL_CONTROLLER_IP
CONTROLLER_PORT_SERVER=$GLOBAL_CONTROLLER_PORT_FOR_SERVER

echo "GPU list: $GPU_LIST"
echo "Local IP: $LOCAL_IP"
echo "Load balancer IP: $LOAD_BALANCER_IP"
echo "Load balancer port: $LOAD_BALANCER_PORT_SERVER"
echo "Controller IP: $CONTROLLER_IP"
echo "Controller port: $CONTROLLER_PORT_SERVER"
echo "Backend: $BACKEND"
echo "World size: $WORLD_SIZE"
sleep 1

PID_SERVER=()

for GPU in $GPU_LIST
do
    echo "Start worker $GPU"
	nohup python ./build/bin/server.py $LOCAL_IP $GPU $LOCAL_IP $CACHE_PORT $LOAD_BALANCER_IP $LOAD_BALANCER_PORT_SERVER $CONTROLLER_IP $CONTROLLER_PORT_SERVER $BACKEND $WORLD_SIZE >tmp/test$TEST_INDEX/$SYSTEM_TYPE/$MODEL_SEED/log_worker_$GPU.txt 2>&1 &
    PID_SERVER+=($!)
    sleep 1
done

# Save the PIDs to a file
echo "${PID_SERVER[@]}" > .pid_server