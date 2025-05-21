#ifndef BALANCE_LB_SERVER_AGENT_H
#define BALANCE_LB_SERVER_AGENT_H

#include <unistd.h>

#include <string>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "utils/utils.h"

#include "base_agent.h"
#include "data_type.h"

class ServerAgent: public BaseAgent {
public:
    static std::shared_ptr<ServerAgent> getServerAgent(uint64_t id) {
        const std::lock_guard<std::mutex> guard(_server_map_lock);
        return _server_map[id];
    }
    static std::shared_ptr<ServerAgent> createServerAgent(std::shared_ptr<balance::util::TcpAgent> agent, uint64_t id = 0) {
        const std::lock_guard<std::mutex> guard(_server_map_lock);
        std::shared_ptr<ServerAgent> sa = std::shared_ptr<ServerAgent>(new ServerAgent(agent, id));
        _server_map[sa->_id] = sa;
        return sa;
    }
private:
    static std::unordered_map<uint64_t, std::shared_ptr<ServerAgent> > _server_map;
    static std::mutex _server_map_lock;

public:
    ~ServerAgent();

    int getIp();
    int getPort();
    void sendRequest(std::shared_ptr<LBTask> request);
    std::shared_ptr<LBTask> recvResponse();

private:
    ServerAgent(std::shared_ptr<balance::util::TcpAgent> agent, uint64_t id);
private:
    int _ip;
    int _port;
    balance::util::AtomicQueue<std::shared_ptr<LBTask> > _queue_for_pending;
};

#endif