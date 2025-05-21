#include <unistd.h>
#include <string.h>

#include <string>
#include <mutex>

#include "utils/utils.h"

#include "client_agent.h"

using namespace std;
using namespace balance::util;

uint64_t _client_id_generator = 0;
mutex _lock;

uint64_t client_generate_id() {
    const lock_guard<mutex> guard(_lock);
    return ++_client_id_generator;
}

ClientAgent::ClientAgent(shared_ptr<TcpAgent> agent):
BaseAgent(agent, client_generate_id()) {}

ClientAgent::~ClientAgent() {}

shared_ptr<LBTask> ClientAgent::recvRequest() {
    // cout << "Receive request, " << _id << ", 1" << endl;
    shared_ptr<LBTask> request(new LBTask());
    // cout << "Receive request, " << _id << ", 2" << endl;
    request->op = LB_TASK_REQUEST;
    // cout << "Receive request, " << _id << ", 3" << endl;
    request->id = getId();
    // cout << "Receive request, " << _id << ", 4" << endl;
    if (ERRNO_SUCCESS != _agent->tcpRecvString(request->model_name))
        return nullptr;
    // cout << "Receive request, " << _id << ", 5" << endl;
    if (ERRNO_SUCCESS != _agent->tcpRecvWithLength(request->data_ptr, request->data_size))
        return nullptr;
    // cout << "Receive request, " << _id << ", 6" << endl;
    return request;
}

void ClientAgent::sendResponse(shared_ptr<LBTask> response) {
    if (ERRNO_SUCCESS != _agent->tcpSendWithLength(response->data_ptr, response->data_size))
        cout << "Reply Error" << endl;
}