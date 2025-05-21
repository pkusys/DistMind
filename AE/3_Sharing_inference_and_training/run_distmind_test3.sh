#!/bin/bash
source settings/config.sh

set -e  # Exit on error

PRIFIX=${1-"tmp/test3"}

# Create log directory
mkdir -p $PRIFIX/distmind

echo "Cleaning up /dev/shm..."
find /dev/shm/* -writable -exec rm -rf {} + 2>/dev/null  || true
sleep 5

# Function to kill all subprocesses
cleanup() {
    # Cleanup all processes
    echo "Cleaning up all background processes..."
    if kill -0 $PID_CLIENT $PID_CLIENT_INFERENCE 2>/dev/null; then
        kill -9 -$PID_CLIENT
        sleep 2
    fi

    for pid in "${SERVER_PIDS[@]}"; do
        if kill -0 $pid 2>/dev/null; then
            kill -9 $pid
            sleep 2
        fi
    done

    for pid in $PID_CACHE $PID_LB $PID_CONTROLLER $PID_STORAGE $PID_METADATA; do
        if kill -0 $pid 2>/dev/null; then
            kill -9 $pid
            sleep 2
        fi
    done

    wait
    echo "All processes finished and cleaned up."

    # Step 11: Stop storage
    find /dev/shm/* -writable -exec rm -rf {} + 2>/dev/null
}

# Trap SIGINT (Ctrl+C) and call cleanup function
trap cleanup SIGINT
trap cleanup EXIT

# Step 1: Generate
echo "Running generate..."
./scripts/run_generate.sh

# Step 2: Metadata
echo "Running metadata..."
METADATA_PORT=$GLOBAL_METADATA_PORT
./build/bin/metadata_storage 0.0.0.0 $METADATA_PORT > $PRIFIX/distmind/log_metadata.txt 2>&1 &
PID_METADATA=$!

# Step 3: Storage
echo "Starting storage..."
STORAGE_PORT=$GLOBAL_STORAGE_PORT
./build/bin/storage $STORAGE_PORT $STORAGE_PORT > $PRIFIX/distmind/log_storage.txt 2>&1 &
PID_STORAGE=$!

sleep 5  # Give storage a bit of head start

# Step 4: Deploy
echo "Running deploy..."
./scripts/run_deploy.sh 

# Step 5: Controller
echo "Starting controller..."
python build/bin/controller.py --config settings/controller.json --use_train True > $PRIFIX/distmind/log_controller.txt 2>&1 &
PID_CONTROLLER=$!

sleep 10  # Give controller a bit of head start

# Step 6: Load Balancer
echo "Starting load balancer..."
PORT_FOR_CLIENT=$GLOBAL_LOAD_BALANCER_PORT_FOR_CLIENT
PORT_FOR_CACHE=$GLOBAL_LOAD_BALANCER_PORT_FOR_CACHE
PORT_FOR_SERVER=$GLOBAL_LOAD_BALANCER_PORT_FOR_SERVER
CONTROLLER_IP=$GLOBAL_CONTROLLER_IP
CONTROLLER_PORT=$GLOBAL_CONTROLLER_PORT_FOR_SUBSCRIBER
METADATA_IP=$GLOBAL_METADATA_IP

echo "Run load_balancer"
echo "Port for client: $PORT_FOR_CLIENT"
echo "Port for cache: $PORT_FOR_CACHE"
echo "Port for servers: $PORT_FOR_SERVER"
echo "Controller IP: $CONTROLLER_IP"
echo "Controller port: $CONTROLLER_PORT"
echo "Metadata IP: $METADATA_IP"
echo "Metadata port: $METADATA_PORT"
echo ""

./build/bin/load_balancer basic 0.0.0.0 $PORT_FOR_CLIENT 0.0.0.0 $PORT_FOR_CACHE 0.0.0.0 $PORT_FOR_SERVER $CONTROLLER_IP $CONTROLLER_PORT $METADATA_IP $METADATA_PORT > $PRIFIX/distmind/log_LB.txt 2>&1 &
PID_LB=$!

sleep 10  # Give load balancer a bit of head start

# Step 7: Cache
echo "Starting cache..."
LOCAL_IP=$GLOBAL_LOCAL_IP
LOAD_BALANCER_IP=$GLOBAL_LOAD_BALANCER_IP

CACHE_SIZE=4096000000
CACHE_BLOCK_SIZE=4096000

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
FI_EFA_ENABLE_SHM_TRANSFER=0 ./build/bin/cache $LOCAL_IP $CACHE_PORT $METADATA_IP $METADATA_PORT $LOAD_BALANCER_IP $LOAD_BALANCER_PORT_CACHE cachecache $CACHE_SIZE $CACHE_BLOCK_SIZE > $PRIFIX/distmind/log_cache.txt 2>&1 &
PID_CACHE=$!

sleep 10  # Give cache a bit of head start

# Step 8: Server
echo "Starting server..."

LOAD_BALANCER_PORT_SERVER=$GLOBAL_LOAD_BALANCER_PORT_FOR_SERVER
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
	nohup python ./build/bin/server.py $LOCAL_IP $GPU $LOCAL_IP $CACHE_PORT $LOAD_BALANCER_IP $LOAD_BALANCER_PORT_SERVER $CONTROLLER_IP $CONTROLLER_PORT_SERVER $BACKEND $WORLD_SIZE >$PRIFIX/distmind/log_worker_$GPU.txt 2>&1 &
    PID_SERVER+=($!)
    sleep 1
done

# Step 9: Wait for readiness
echo "Waiting 120 seconds before running client..."
sleep 120

# Step 10: Run client
echo "Running client train..."
setsid python build/bin/client_max_train.py $CONTROLLER_IP $CONTROLLER_PORT $LOAD_BALANCER_IP $PORT_FOR_CLIENT 0 > $PRIFIX/distmind/log_train.txt 2>&1 &
PID_CLIENT=$!

echo "Running client inference..."
setsid python build/bin/client_max_inference.py $CONTROLLER_IP $CONTROLLER_PORT $LOAD_BALANCER_IP $PORT_FOR_CLIENT 0 > $PRIFIX/distmind/log_client.txt 2>&1 &
PID_CLIENT_INFERENCE=$!

sleep 400

