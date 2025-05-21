#!/bin/bash
source settings/config.sh

CLIENT=$1

echo "Starting client $CLIENT..."
./build/bin/shard_cli $CLIENT $GLOBAL_CONTROLLER_IP $GLOBAL_CONTROLLER_PORT_FOR_SUBSCRIBER 40 > tmp/test9/log_client_$CLIENT.txt 2>&1 &

echo $! > .pid_storage_client