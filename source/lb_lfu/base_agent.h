#ifndef BALANCE_LB_BASE_AGENT_H
#define BALANCE_LB_BASE_AGENT_H

#include <unistd.h>

#include <string>
#include <memory>
#include <iostream>

#include "utils/utils.h"

#include "data_type.h"

class BaseAgent {
public:
    BaseAgent(std::shared_ptr<balance::util::TcpAgent> agent, uint64_t id = 0):
    _agent(agent), _id(id == 0? (uint64_t)_agent.get(): id) {}
    ~BaseAgent() {}
    uint64_t getId() {return _id;}

protected:
    std::shared_ptr<balance::util::TcpAgent> _agent;
    uint64_t _id;
};

#endif