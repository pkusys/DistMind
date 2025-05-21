#ifndef BALANCE_LB_CACHE_AGENT_H
#define BALANCE_LB_CACHE_AGENT_H

#include <unistd.h>

#include <string>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <iostream>

#include "utils/utils.h"

#include "base_agent.h"

class CacheAgent: public BaseAgent {
public:
    static std::shared_ptr<CacheAgent> getCacheAgent(uint64_t id) {
        const std::lock_guard<std::mutex> guard(_cache_map_lock);
        return _cache_map[id];
    }
    static std::shared_ptr<CacheAgent> createCacheAgent(std::shared_ptr<balance::util::TcpAgent> agent, uint64_t id = 0) {
        const std::lock_guard<std::mutex> guard(_cache_map_lock);
        std::shared_ptr<CacheAgent> ca = std::shared_ptr<CacheAgent>(new CacheAgent(agent, id));
        _cache_map[ca->_id] = ca;
        return ca;
    }
private:
    static std::unordered_map<uint64_t, std::shared_ptr<CacheAgent> > _cache_map;
    static std::mutex _cache_map_lock;

public:
    ~CacheAgent();

    int getIp();
    size_t getCapacity();
    void cacheIn(std::string model_name);
    void cacheOut(std::string model_name);

private:
    CacheAgent(std::shared_ptr<balance::util::TcpAgent> agent, uint64_t id);
private:
    int _ip;
    size_t _capacity;
};

#endif