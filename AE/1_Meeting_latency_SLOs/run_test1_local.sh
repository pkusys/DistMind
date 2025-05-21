#!/bin/bash
source settings/config.sh

set -e  # Exit on error

mkdir -p tmp
rm -rf tmp/test1
mkdir -p tmp/test1

MODEL_SEEDS=("res" "inc" "den" "bert" "gpt")
CACHED_SIZES=(15 40 50 8 6)
REMOTE_SIZES=(128 128 128 64 64)

# in gpu
mkdir -p tmp/test1/gpu

# Run gpu tests for each model seed with corresponding size
echo "Running gpu tests for different model seeds..."
for i in "${!MODEL_SEEDS[@]}"; do
    SEED=${MODEL_SEEDS[$i]}
    SIZE=1
    
    echo "======================================================================================"
    echo "Running gpu test for seed: $SEED with size: $SIZE"
    echo "======================================================================================"
    
    # Run the test for this seed and size
    ./AE/1_Meeting_latency_SLOs/run_distmind_test1.sh $SIZE $SEED "gpu"
    
    # Wait a bit between runs to ensure clean shutdown
    sleep 10
done

# in cache
mkdir -p tmp/test1/distmind_cache

# Run distmind_cache tests for each model seed with corresponding size
echo "Running distmind_cache tests for different model seeds..."
for i in "${!MODEL_SEEDS[@]}"; do
    SEED=${MODEL_SEEDS[$i]}
    SIZE=${CACHED_SIZES[$i]}
    
    echo "======================================================================================"
    echo "Running distmind_cache test for seed: $SEED with size: $SIZE"
    echo "======================================================================================"
    
    # Run the test for this seed and size
    ./AE/1_Meeting_latency_SLOs/run_distmind_test1.sh $SIZE $SEED "distmind_cache"
    
    # Wait a bit between runs to ensure clean shutdown
    sleep 10
done

# distmind_remote
mkdir -p tmp/test1/distmind_remote

# Run distmind_remote tests for each model seed with corresponding size
echo "Running distmind_remote tests for different model seeds..."
for i in "${!MODEL_SEEDS[@]}"; do
    SEED=${MODEL_SEEDS[$i]}
    SIZE=${REMOTE_SIZES[$i]}
    
    echo "======================================================================================"
    echo "Running distmind_remote test for seed: $SEED with size: $SIZE"
    echo "======================================================================================"
    
    # Run the test for this seed and size
    ./AE/1_Meeting_latency_SLOs/run_distmind_test1.sh $SIZE $SEED "distmind_remote"
    
    # Wait a bit between runs to ensure clean shutdown
    sleep 10
done


# mps
mkdir -p tmp/test1/mps

# Run mps tests for each model seed with corresponding size
echo "Running mps tests for different model seeds..."
for i in "${!MODEL_SEEDS[@]}"; do
    SEED=${MODEL_SEEDS[$i]}
    SIZE=${REMOTE_SIZES[$i]}
    
    echo "======================================================================================"
    echo "Running mps test for seed: $SEED with size: $SIZE"
    echo "======================================================================================"
    
    # Run the test for this seed and size
    ./AE/1_Meeting_latency_SLOs/run_mps_test1.sh $SIZE $SEED
    
    # Wait a bit between runs to ensure clean shutdown
    sleep 10
done

# ray
mkdir -p tmp/test1/ray

# Run ray tests for each model seed with corresponding size
echo "Running ray tests for different model seeds..."
for i in "${!MODEL_SEEDS[@]}"; do
    SEED=${MODEL_SEEDS[$i]}
    SIZE=${REMOTE_SIZES[$i]}
    
    echo "======================================================================================"
    echo "Running ray test for seed: $SEED with size: $SIZE"
    echo "======================================================================================"
    
    # Run the test for this seed and size
    ./AE/1_Meeting_latency_SLOs/run_ry_test1.sh $SIZE $SEED
    
    # Wait a bit between runs to ensure clean shutdown
    sleep 10
done

# end of script
echo "All tests completed successfully."