#!/bin/bash
source settings/config.sh

set -e  # Exit on error

mkdir -p tmp
python ./source/py_utils/launch_remote.py --launch_part create_dir  --log_dir tmp --test_index 2
rm -rf tmp/test2
python ./source/py_utils/launch_remote.py --launch_part create_dir  --log_dir tmp/test2 --test_index 2 --stop
mkdir -p tmp/test2
python ./source/py_utils/launch_remote.py --launch_part create_dir  --log_dir tmp/test2 --test_index 2


# in gpu
mkdir -p tmp/test2/gpu
python ./source/py_utils/launch_remote.py --launch_part create_dir  --log_dir tmp/test2/gpu --test_index 2

echo "======================================================================================"
echo "Running gpu test"
echo "======================================================================================"
    
# Run the test for this seed and size
./AE/2_End-to-end_performance/run_distmind_test2_remote.sh 1 "res" "gpu"
    
# Wait a bit between runs to ensure clean shutdown
sleep 10

# distmind
mkdir -p tmp/test2/distmind
python ./source/py_utils/launch_remote.py --launch_part create_dir  --log_dir tmp/test2/distmind --test_index 2

# Run distmind tests for each model seed with corresponding size
echo "======================================================================================"
echo "Running distmind test"
echo "======================================================================================"
    
# Run the test for this seed and size
./AE/2_End-to-end_performance/run_distmind_test2_remote.sh $GLOBAL_NUM_MODELS "res" "distmind"
    
# Wait a bit between runs to ensure clean shutdown
sleep 10

# mps
mkdir -p tmp/test2/mps
python ./source/py_utils/launch_remote.py --launch_part create_dir  --log_dir tmp/test2/mps --test_index 2

# Run mps tests for each model seed with corresponding size
echo "======================================================================================"
echo "Running mps test"
echo "======================================================================================"
    
# Run the test for this seed and size
./AE/2_End-to-end_performance/run_mps_test2_remote.sh
    
# Wait a bit between runs to ensure clean shutdown
sleep 10

# ray
mkdir -p tmp/test2/ray
python ./source/py_utils/launch_remote.py --launch_part create_dir  --log_dir tmp/test2/ray --test_index 2

# Run ray tests for each model seed with corresponding size
echo "======================================================================================"
echo "Running ray test"
echo "======================================================================================"
    
# Run the test for this seed and size
./AE/2_End-to-end_performance/run_ry_test2_remote.sh
    
# Wait a bit between runs to ensure clean shutdown
sleep 10

# end of script
echo "All tests completed successfully."