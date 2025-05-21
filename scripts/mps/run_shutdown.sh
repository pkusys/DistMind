python source/mps/shutdown_mps_daemon.py

find /dev/shm/* -writable -exec rm -rf {} + 2>/dev/null
