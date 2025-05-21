#ifndef BALANCE_LB_CLIENT_AGENT_H
#define BALANCE_LB_CLIENT_AGENT_H

#include <unistd.h>

#include <string>
#include <memory>
#include <mutex>

#include "utils/utils.h"

#include "base_agent.h"
#include "data_type.h"

uint64_t client_generate_id();

class ClientAgent: public BaseAgent {
public:
    ClientAgent(std::shared_ptr<balance::util::TcpAgent> agent);
    ~ClientAgent();

    std::shared_ptr<LBTask> recvRequest();
    void sendResponse(std::shared_ptr<LBTask> response);
};

#endif