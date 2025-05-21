#!/bin/bash

if [ -f .pid_cache ]; then
    PID=$(cat .pid_cache)
    if ps -p $PID > /dev/null; then
        echo "Killing cache process $PID"
        kill $PID
    else
        echo "Cache process $PID no longer exists"
    fi
    rm .pid_cache
    echo "Cache process killed"
else
    echo "No PID file found (.pid_cache)"
fi

if [ -f .pid_server ]; then
    PIDs=$(cat .pid_server)
    
    for PID in $PIDs; do
        if ps -p $PID > /dev/null; then
            echo "Killing process $PID"
            kill $PID
        else
            echo "Process $PID no longer exists"
        fi
    done
    
    rm .pid_server
    echo "All server processes killed"
else
    echo "No PID file found (.pid_server)"
fi