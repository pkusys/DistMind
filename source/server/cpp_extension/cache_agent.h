#ifndef BALANCE_SERVER_CACHE_AGENT_H
#define BALANCE_SERVER_CACHE_AGENT_H

#include <unistd.h>

#include <string>
#include <memory>
#include <vector>

#include "utils/tcp/tcp.h"
#include "utils/shared_memory/shared_memory.h"
#include "utils/memory_manager/memory_manager.h"

class CacheAgent {
public:
    CacheAgent(std::string addr, int port);
    ~CacheAgent();
    char* getShmPointer();
    size_t getShmSize();
    std::shared_ptr<balance::util::BlockGroup> get(std::string key);
private:
    std::shared_ptr<balance::util::TcpClient> _client;
    std::shared_ptr<balance::util::SharedMemory> _shm;
};

#endif