#!/bin/bash
source settings/config.sh

set -e  # Exit on error

PRIFIX=${1-"tmp/test3"}
# Create log directory
mkdir -p $PRIFIX/gpu
python ./source/py_utils/launch_remote.py --launch_part create_dir  --log_dir $PRIFIX/gpu --test_index 3
sleep 5

echo "Cleaning up /dev/shm..."
python ./source/py_utils/launch_remote.py --launch_part cleanup --test_index 3
sleep 5

# Function to kill all subprocesses
cleanup() {
    # Cleanup all processes
    echo "Cleaning up all background processes..."
    python ./source/py_utils/launch_remote.py --launch_part dist_server --test_index 3 --stop 
    sleep 5

    python ./source/py_utils/launch_remote.py --launch_part dist_storage --test_index 3 --stop
    sleep 5

    for pid in $PID_CLIENT_INFERENCE $PID_LB $PID_CONTROLLER $PID_METADATA; do
        if kill -0 $pid 2>/dev/null; then
            kill -9 $pid
            sleep 2
        fi
    done

    wait
    echo "All processes finished and cleaned up."

    # Step 11: Stop storage
    python ./source/py_utils/launch_remote.py --launch_part cleanup --test_index 3
}

# Trap SIGINT (Ctrl+C) and call cleanup function
trap cleanup SIGINT
trap cleanup EXIT

# Step 1: Generate
echo "Running generate..."
./scripts/run_generate.sh 1

# Step 2: Metadata
echo "Running metadata..."
METADATA_PORT=$GLOBAL_METADATA_PORT
./build/bin/metadata_storage 0.0.0.0 $METADATA_PORT > $PRIFIX/gpu/log_metadata.txt 2>&1 &
PID_METADATA=$!

# Step 3: Storage
echo "Starting storage..."
python ./source/py_utils/launch_remote.py --launch_part dist_storage --test_index 3 --log_file $PRIFIX/gpu/log_storage.txt
sleep 5  # Give storage a bit of head start

# Step 4: Deploy
echo "Running deploy..."
./scripts/run_deploy.sh 

# Step 5: Controller
echo "Starting controller..."
python build/bin/controller.py --config settings/controller.json > $PRIFIX/gpu/log_controller.txt 2>&1 &
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

./build/bin/load_balancer basic 0.0.0.0 $PORT_FOR_CLIENT 0.0.0.0 $PORT_FOR_CACHE 0.0.0.0 $PORT_FOR_SERVER $CONTROLLER_IP $CONTROLLER_PORT $METADATA_IP $METADATA_PORT > $PRIFIX/gpu/log_LB.txt 2>&1 &
PID_LB=$!

sleep 10  # Give load balancer a bit of head start

# Step 7: Server Side
echo "Starting server side..."
python ./source/py_utils/launch_remote.py --launch_part dist_server --test_index 3 --n_models $GLOBAL_NUM_MODELS --model_seed res --system_type gpu

# Step 9: Wait for readiness
echo "Waiting 120 seconds before running client..."
sleep 120

# Step 10: Run client
echo "Running client inference..."
setsid python build/bin/client_max_inference.py $CONTROLLER_IP $CONTROLLER_PORT $LOAD_BALANCER_IP $LOAD_BALANCER_PORT 0 > $PRIFIX/gpu/log_client.txt 2>&1 &
PID_CLIENT_INFERENCE=$!

sleep 400

