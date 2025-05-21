source settings/config.sh

PORT_FOR_CLIENT=$GLOBAL_LOAD_BALANCER_PORT_FOR_CLIENT
PORT_FOR_SERVER=$GLOBAL_LOAD_BALANCER_PORT_FOR_SERVER
CONTROLLER_IP=$GLOBAL_CONTROLLER_IP
CONTROLLER_PORT=$GLOBAL_CONTROLLER_PORT_FOR_SUBSCRIBER

echo "Controller IP: $CONTROLLER_IP"
echo "Controller port: $CONTROLLER_PORT"
echo "Port for client: $PORT_FOR_CLIENT"
echo "Port for servers: $PORT_FOR_SERVER"

python source/mps/load_balancer.py --controller-address $CONTROLLER_IP --controller-port $CONTROLLER_PORT --client-port  $PORT_FOR_CLIENT --server-port  $PORT_FOR_SERVER --fill-train
echo "Run load_balancer" 