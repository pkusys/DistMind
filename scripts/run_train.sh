source settings/config.sh

LOCAL_IP=$GLOBAL_LOCAL_IP
GPU_INDEX_BASE=$GLOBAL_SERVER_PORT_FOR_INDEX
TRAIN_CONTROLLER_IP=$GLOBAL_TRAIN_CONTROLLER_IP
TRAIN_CONTROLLER_PORT=$GLOBAL_TRAIN_CONTROLLER_PORT_FOR_TRAINING
TRAINING_PORT=$GLOBAL_TRAINING_PORT

python build/bin/train.py  -a resnet152 --dist-url "tcp://127.0.0.1:$TRAINING_PORT" --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --local-ip $LOCAL_IP --gpu-index-base $GPU_INDEX_BASE --train-controller-ip $TRAIN_CONTROLLER_IP --train-controller-port $TRAIN_CONTROLLER_PORT none