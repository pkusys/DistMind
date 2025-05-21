#ifndef BALANCE_SERVER_LB_AGENT_H
#define BALANCE_SERVER_LB_AGENT_H

#include <unistd.h>

#include <string>
#include <memory>
#include <vector>

#include "utils/tcp/tcp.h"

class LBAgent {
public:
    LBAgent(std::string lb_addr, int lb_port, std::string addr_for_cli, int port_for_cli);
    ~LBAgent();
    void completeTask(std::string model_name, char* data, size_t size);
    std::string getTask();
    std::pair<std::shared_ptr<char>, size_t> getData();
private:
    std::shared_ptr<balance::util::TcpClient> _client;
};

#endif