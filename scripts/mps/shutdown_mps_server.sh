#!/bin/bash

if [ -f .pid_mps_server ]; then
    PID=$(cat .pid_mps_server)
    
    if ps -p $PID > /dev/null; then
        echo "Killing MPS server process $PID"
        kill -9 -$PID
    else
        echo "MPS server process $PID no longer exists"
    fi
    rm .pid_mps_server
    echo "All server processes killed"
else
    echo "No PID file found (.pid_mps_server)"
fi