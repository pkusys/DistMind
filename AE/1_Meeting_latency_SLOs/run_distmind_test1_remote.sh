#!/bin/bash
source settings/config.sh

set -e  # Exit on error

# Set default values for parameters
# $1: NUM_MODELS - default is GLOBAL_NUM_MODELS from config.sh
# $2: MODEL_SEED - default is 'res' (for resnet)
NUM_MODELS=${1:-$GLOBAL_NUM_MODELS}
MODEL_SEED=${2:-"res"}
SYSTEM_TYPE=${3:-"distmind_remote"}

# Create log directory
mkdir -p tmp/test1/$SYSTEM_TYPE/$MODEL_SEED
python ./source/py_utils/launch_remote.py --launch_part create_dir  --log_dir tmp/test1/$SYSTEM_TYPE/$MODEL_SEED --test_index 1
sleep 5

echo "Cleaning up /dev/shm..."
python ./source/py_utils/launch_remote.py --launch_part cleanup --test_index 1
sleep 5

# Function to kill all subprocesses
cleanup() {
    # Cleanup all processes
    echo "Cleaning up all background processes..."
    python ./source/py_utils/launch_remote.py --launch_part dist_server --test_index 1 --stop 
    sleep 5

    python ./source/py_utils/launch_remote.py --launch_part dist_storage --test_index 1 --stop
    sleep 5

    for pid in $PID_LB $PID_CONTROLLER $PID_METADATA; do
        if kill -0 $pid 2>/dev/null; then
            kill -9 $pid
            sleep 2
        fi
    done

    wait
    echo "All processes finished and cleaned up."

    # Step 11: Stop storage
    python ./source/py_utils/launch_remote.py --launch_part cleanup --test_index 1
}

# Trap SIGINT (Ctrl+C) and call cleanup function
trap cleanup SIGINT
trap cleanup EXIT

# Step 1: Generate
echo "Running generate..."
./scripts/run_generate.sh $NUM_MODELS $MODEL_SEED

# Step 2: Metadata
echo "Running metadata..."
METADATA_PORT=$GLOBAL_METADATA_PORT
./build/bin/metadata_storage 0.0.0.0 $METADATA_PORT > tmp/test1/$SYSTEM_TYPE/$MODEL_SEED/log_metadata.txt 2>&1 &
PID_METADATA=$!

# Step 3: Storage
echo "Starting storage..."
python ./source/py_utils/launch_remote.py --launch_part dist_storage --test_index 1 --log_file tmp/test1/$SYSTEM_TYPE/$MODEL_SEED/log_storage.txt
sleep 5  # Give storage a bit of head start

# Step 4: Deploy
echo "Running deploy..."
./scripts/run_deploy.sh 

# Step 5: Controller
echo "Starting controller..."
python build/bin/controller.py --config settings/controller.json > tmp/test1/$SYSTEM_TYPE/$MODEL_SEED/log_controller.txt 2>&1 &
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

./build/bin/load_balancer basic 0.0.0.0 $PORT_FOR_CLIENT 0.0.0.0 $PORT_FOR_CACHE 0.0.0.0 $PORT_FOR_SERVER $CONTROLLER_IP $CONTROLLER_PORT $METADATA_IP $METADATA_PORT > tmp/test1/$SYSTEM_TYPE/$MODEL_SEED/log_LB.txt 2>&1 &
PID_LB=$!

sleep 10  # Give load balancer a bit of head start

# Step 7: Server Side
echo "Starting server side..."
python ./source/py_utils/launch_remote.py --launch_part dist_server --test_index 1 --n_models $NUM_MODELS --model_seed $MODEL_SEED --system_type $SYSTEM_TYPE 

# Step 9: Wait for readiness
echo "Waiting 120 seconds before running client..."
sleep 120

# Step 10: Run client
echo "Running client inference..."
./scripts/run_client_max_inference.sh > tmp/test1/$SYSTEM_TYPE/$MODEL_SEED/log_client.txt

