source settings/config.sh

IP_FOR_LOAD_BALANCER=$GLOBAL_LOAD_BALANCER_IP
PORT_FOR_SERVER=$GLOBAL_LOAD_BALANCER_PORT_FOR_SERVER
LOCAL_IP=$GLOBAL_LOCAL_IP
CONTROLLER_PORT=$GLOBAL_CONTROLLER_PORT_FOR_SERVER
CONTROLLER_IP=$GLOBAL_CONTROLLER_IP

WORLD_SIZE=2

echo "IP for client: $IP_FOR_LOAD_BALANCER"
echo "Port for servers: $PORT_FOR_SERVER"
echo "Local IP: $LOCAL_IP"
echo "Controller IP: $CONTROLLER_IP"
echo "Controller port: $CONTROLLER_PORT"

python source/mps/server_agent.py --lb-port $PORT_FOR_SERVER --lb-ip $IP_FOR_LOAD_BALANCER --size-list build/mps/model_sizes.txt --gpu-num $WORLD_SIZE --ctrl-ip $CONTROLLER_IP --ctrl-port $CONTROLLER_PORT --local-ip $LOCAL_IP
