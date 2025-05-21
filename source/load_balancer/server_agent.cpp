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
    cout << "Accept server: " << _id << ", " << (unsigned int)_ip << ", " << _port << endl;
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
    request->time_forward_0 = time_now();
    _agent->tcpSendString(request->model_name);
    _agent->tcpSendWithLength(request->data_ptr, request->data_size);
    _queue_for_pending.push(request);
    request->time_forward = time_now();
    // _queue_for_time.push(time_now());
    // cout << "\t\t\t\tServerAgent::sendRequest: " << getId() << ", " << request->id << endl;
}

shared_ptr<LBTask> ServerAgent::recvResponse() {
    while (_queue_for_pending.empty())
        usleep(10);
    shared_ptr<LBTask> response = _queue_for_pending.pop();
    response->op = LB_TASK_RESPONSE;
    _agent->tcpRecvWithLength(response->data_ptr, response->data_size);

    // double start_time = _queue_for_time.pop();
    // double end_time = time_now();
    // cout << fixed << 1000 * (end_time - start_time) << endl;
    response->time_compute = time_now();
    return response;
}