#!/bin/bash
source settings/config.sh

set -e  # Exit on error

mkdir -p tmp
python ./source/py_utils/launch_remote.py --launch_part create_dir  --log_dir tmp --test_index 3
mkdir -p tmp/test3
python ./source/py_utils/launch_remote.py --launch_part create_dir  --log_dir tmp/test3 --test_index 3
mkdir -p tmp/test3/bound
python ./source/py_utils/launch_remote.py --launch_part create_dir  --log_dir tmp/test3/bound --test_index 3

# mps

echo "======================================================================================"
echo "Running mps test"
echo "======================================================================================"

./AE/3_Sharing_inference_and_training/run_mps_test3_bound_remote.sh

sleep 10

# ray

echo "======================================================================================"
echo "Running ray test"
echo "======================================================================================"

./AE/3_Sharing_inference_and_training/run_ry_test3_bound_remote.sh

sleep 10

# distmind

echo "======================================================================================"
echo "Running distmind test"
echo "======================================================================================"

./AE/3_Sharing_inference_and_training/run_distmind_test3_bound_remote.sh

sleep 10

echo "All tests completed successfully!"

python ./source/py_utils/launch_remote.py --launch_part get_output --log_dir tmp/test3/bound/mps --test_index 3
echo "All output files copied successfully!"
