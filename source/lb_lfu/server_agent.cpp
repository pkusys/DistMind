#include <unistd.h>
#include <string.h>

#include <string>
#include <iostream>

#include "utils/utils.h"

#include "server_agent.h"

using namespace std;
using namespace balance::util;

ServerAgent::ServerAgent(shared_ptr<TcpAgent> agent, uint64_t id):
BaseAgent(agent, id) {
    struct {
        int ip_int;
        int port;
    } server_location;
    agent->tcpRecv((char*)&server_location, sizeof(server_location));

    _ip = server_location.ip_int;
    _port = server_location.port;
    _id = (((uint64_t)_ip) << 32) + (uint64_t)_port;
}

ServerAgent::~ServerAgent() {}

std::unordered_map<uint64_t, std::shared_ptr<ServerAgent> > ServerAgent::_server_map;
std::mutex ServerAgent::_server_map_lock;

int ServerAgent::getIp() {
    return _ip;
}

int ServerAgent::getPort() {
    return _port;
}

void ServerAgent::sendRequest(shared_ptr<LBTask> request) {
    _agent->tcpSendString(request->model_name);
    _agent->tcpSendWithLength(request->data_ptr, request->data_size);
    _queue_for_pending.push(request);
    // cout << "\t\t\t\tServerAgent::sendRequest: " << getId() << ", " << request->id << endl;
}

shared_ptr<LBTask> ServerAgent::recvResponse() {
    shared_ptr<LBTask> response(new LBTask());
    response->op = LB_TASK_RESPONSE;
    _agent->tcpRecvWithLength(response->data_ptr, response->data_size);

    shared_ptr<LBTask> request = _queue_for_pending.pop();
    response->id = request->id;
    response->model_name = request->model_name;
    

    return response;
}