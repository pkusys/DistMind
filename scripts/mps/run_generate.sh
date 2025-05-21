source settings/config.sh

# Set default values for parameters
# $1: NUM_MODELS - default is GLOBAL_NUM_MODELS from config.sh
# $2: MODEL_SEED - default is 'res' (for resnet)
NUM_MODELS=${1:-$GLOBAL_NUM_MODELS}
MODEL_SEED=${2:-"res"}
DISTRIBUTION_RANK=1

mkdir -p build/mps

python build/bin/generate_model_list.py modelsettings/model_list_seed_${MODEL_SEED}.txt $NUM_MODELS $DISTRIBUTION_RANK build/mps/model_list.txt
echo "Generate model list"
sleep 1

echo "Complete"
