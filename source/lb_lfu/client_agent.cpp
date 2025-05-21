#include <unistd.h>
#include <string.h>

#include <string>

#include "utils/utils.h"

#include "client_agent.h"

using namespace std;
using namespace balance::util;

ClientAgent::ClientAgent(shared_ptr<TcpAgent> agent, uint64_t id):
BaseAgent(agent, id) {}

ClientAgent::~ClientAgent() {}

shared_ptr<LBTask> ClientAgent::recvRequest() {
    shared_ptr<LBTask> request(new LBTask());
    request->op = LB_TASK_REQUEST;
    request->id = getId();
    _agent->tcpRecvString(request->model_name);
    _agent->tcpRecvWithLength(request->data_ptr, request->data_size);
    return request;
}

void ClientAgent::sendResponse(shared_ptr<LBTask> response) {
    _agent->tcpSendWithLength(response->data_ptr, response->data_size);
}