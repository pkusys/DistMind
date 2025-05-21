#ifndef BALANCE_TCP_PATTERN_H
#define BALANCE_TCP_PATTERN_H

#include <unistd.h>

#include <string>
#include <memory>
#include <thread>

#include "utils/tcp/tcp.h"

namespace balance {
namespace util {

struct AgentHandler;
class TcpServerParallelWithAgentHandler;

struct AgentHandler {
    virtual void run(std::shared_ptr<TcpAgent> agent) = 0;
};

class TcpServerParallelWithAgentHandler {
public:
    TcpServerParallelWithAgentHandler(std::string address, int port, std::shared_ptr<AgentHandler> handler);
    ~TcpServerParallelWithAgentHandler();
private:
    std::shared_ptr<TcpServer> _server;
    std::shared_ptr<std::thread> _listener;
    std::shared_ptr<AgentHandler> _handler;
};

} //namespace util
} //namespace balance

#endif