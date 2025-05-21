#!/bin/bash
source settings/config.sh

set -e  # Exit on error

mkdir -p tmp
mkdir -p tmp/test3
mkdir -p tmp/test3/bound

# mps

echo "======================================================================================"
echo "Running mps test"
echo "======================================================================================"

./AE/3_Sharing_inference_and_training/run_mps_test3_bound.sh

sleep 10

# ray

echo "======================================================================================"
echo "Running ray test"
echo "======================================================================================"

./AE/3_Sharing_inference_and_training/run_ry_test3_bound.sh

sleep 10

# distmind

echo "======================================================================================"
echo "Running distmind test"
echo "======================================================================================"

./AE/3_Sharing_inference_and_training/run_distmind_test3_bound.sh

sleep 10

echo "All tests completed successfully!"