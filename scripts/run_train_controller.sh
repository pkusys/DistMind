source settings/config.sh

TRAIN_CONTROLLER_IP=$GLOBAL_TRAIN_CONTROLLER_IP
TRAIN_CONTROLLER_PORT=$GLOBAL_TRAIN_CONTROLLER_PORT_FOR_TRAINING
CONTROLLER_IP=$GLOBAL_CONTROLLER_IP
CONTROLLER_PORT=$GLOBAL_CONTROLLER_PORT_FOR_SUBSCRIBER

python build/bin/train_controller.py $TRAIN_CONTROLLER_IP $TRAIN_CONTROLLER_PORT $CONTROLLER_IP $CONTROLLER_PORT