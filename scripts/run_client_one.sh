source settings/config.sh

LOAD_BALANCER_IP=$GLOBAL_LOAD_BALANCER_IP
LOAD_BALANCER_PORT=$GLOBAL_LOAD_BALANCER_PORT_FOR_CLIENT

python build/bin/client_one.py build/resource/request_list.txt $LOAD_BALANCER_IP $LOAD_BALANCER_PORT 