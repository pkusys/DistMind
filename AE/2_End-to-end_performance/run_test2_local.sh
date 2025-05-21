#!/bin/bash
source settings/config.sh

set -e  # Exit on error

mkdir -p tmp
rm -rf tmp/test2
mkdir -p tmp/test2


# in gpu
mkdir -p tmp/test2/gpu

echo "======================================================================================"
echo "Running gpu test"
echo "======================================================================================"
    
# Run the test for this seed and size
./AE/2_End-to-end_performance/run_distmind_test2.sh 1 "res" "gpu"
    
# Wait a bit between runs to ensure clean shutdown
sleep 10

# distmind
mkdir -p tmp/test2/distmind

# Run distmind tests for each model seed with corresponding size
echo "======================================================================================"
echo "Running distmind test"
echo "======================================================================================"
    
# Run the test for this seed and size
./AE/2_End-to-end_performance/run_distmind_test2.sh $GLOBAL_NUM_MODELS "res" "distmind"
    
# Wait a bit between runs to ensure clean shutdown
sleep 10

# mps
mkdir -p tmp/test2/mps

# Run mps tests for each model seed with corresponding size
echo "======================================================================================"
echo "Running mps test"
echo "======================================================================================"
    
# Run the test for this seed and size
./AE/2_End-to-end_performance/run_mps_test2.sh
    
# Wait a bit between runs to ensure clean shutdown
sleep 10

# ray
mkdir -p tmp/test2/ray

# Run ray tests for each model seed with corresponding size
echo "======================================================================================"
echo "Running ray test"
echo "======================================================================================"
    
# Run the test for this seed and size
./AE/2_End-to-end_performance/run_ry_test2.sh
    
# Wait a bit between runs to ensure clean shutdown
sleep 10

# end of script
echo "All tests completed successfully."