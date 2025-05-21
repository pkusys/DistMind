#!/bin/bash
source settings/config.sh

set -e  # Exit on error

mkdir -p tmp
rm -rf tmp/test3
mkdir -p tmp/test3

# in gpu
mkdir -p tmp/test3/gpu

echo "======================================================================================"
echo "Running gpu test"
echo "======================================================================================"

./AE/3_Sharing_inference_and_training/run_gpu_test3.sh

sleep 10

# mps
mkdir -p tmp/test3/mps

echo "======================================================================================"
echo "Running mps test"
echo "======================================================================================"

./AE/3_Sharing_inference_and_training/run_mps_test3.sh

sleep 10

# ray
mkdir -p tmp/test3/ray

echo "======================================================================================"
echo "Running ray test"
echo "======================================================================================"

./AE/3_Sharing_inference_and_training/run_ry_test3.sh

sleep 10

# distmind
mkdir -p tmp/test3/distmind

echo "======================================================================================"
echo "Running distmind test"
echo "======================================================================================"

./AE/3_Sharing_inference_and_training/run_distmind_test3.sh

sleep 10

echo "All tests completed successfully!"