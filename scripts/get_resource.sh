# !/bin/bash
source settings/config.sh

PORTS=(
    $GLOBAL_CONTROLLER_PORT_FOR_SERVER
    $GLOBAL_CONTROLLER_PORT_FOR_SUBSCRIBER
    $GLOBAL_TRAIN_CONTROLLER_PORT_FOR_TRAINING
    $GLOBAL_LOAD_BALANCER_PORT_FOR_CACHE
    $GLOBAL_LOAD_BALANCER_PORT_FOR_SERVER
    $GLOBAL_LOAD_BALANCER_PORT_FOR_CLIENT
    $GLOBAL_CAHCE_PORT_FOR_SERVER
    $GLOBAL_TRAINING_PORT
    $GLOBAL_STORAGE_PORT
    $GLOBAL_METADATA_PORT
)

# kill processes using the specified ports
for PORT in "${PORTS[@]}"; do
    echo "Killing process using port $PORT..."
    PID=$(lsof -t -i:$PORT)
    if [ -n "$PID" ]; then
        kill -9 $PID
        echo "Killed process $PID using port $PORT"
    else
        echo "No process found using port $PORT"
    fi
done

