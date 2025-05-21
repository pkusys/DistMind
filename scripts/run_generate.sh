source settings/config.sh

# Set default values for parameters
# $1: NUM_MODELS - default is GLOBAL_NUM_MODELS from config.sh
# $2: MODEL_SEED - default is 'res' (for resnet)
NUM_MODELS=${1:-$GLOBAL_NUM_MODELS}
MODEL_SEED=${2:-"res"}
BATCH_SIZE=${3:-32768000}
DISTRIBUTION_RANK=1

mkdir -p build/resource

python build/bin/generate_model_list.py modelsettings/model_list_seed_${MODEL_SEED}.txt $NUM_MODELS $DISTRIBUTION_RANK build/resource/model_list.txt
echo "Generate model list"
sleep 1

python build/bin/generate_model_distribution.py settings/storage_list.txt build/resource/model_list.txt build/resource/model_distribution.txt
echo "Generate model distribution"
sleep 1

python build/bin/generate_file.py build/resource/model_list.txt build/resource/kv.bin $BATCH_SIZE
echo "Generate binary files for models"

echo "Complete"