#include <unistd.h>
#include <string.h>

#include <string>
#include <iostream>

#include "utils/utils.h"

#include "cache_agent.h"

using namespace std;
using namespace balance::util;

CacheAgent::CacheAgent(shared_ptr<TcpAgent> agent, uint64_t id):
BaseAgent(agent, id) {
    struct {
        int ip_int;
        int port;
        size_t capacity;
    } cache_info;
    agent->tcpRecv((char*)&cache_info, sizeof(cache_info));

    _ip = cache_info.ip_int;
    _id = (((uint64_t)_ip) << 32);
    _capacity = cache_info.capacity;
}

CacheAgent::~CacheAgent() {}

std::unordered_map<uint64_t, std::shared_ptr<CacheAgent> > CacheAgent::_cache_map;
std::mutex CacheAgent::_cache_map_lock;

int CacheAgent::getIp() {
    return _ip;
}

size_t CacheAgent::getCapacity() {
    return _capacity;
}

void CacheAgent::cacheOut(string model_name) {
    _agent->tcpSend((char*)&SIGNAL_CACHE_OUT, sizeof(SIGNAL_CACHE_OUT));
    _agent->tcpSendString(model_name);
    cout << "CacheAgent::cacheOut: " << model_name << endl;
}

void CacheAgent::cacheIn(string model_name) {
    _agent->tcpSend((char*)&SIGNAL_CACHE_IN, sizeof(SIGNAL_CACHE_IN));
    _agent->tcpSendString(model_name);
    cout << "CacheAgent::cacheIn: " << model_name << endl;
    
    int signal_reply = SIGNAL_CACHE_REPLY;
    _agent->tcpRecv((char*)&signal_reply, sizeof(signal_reply));
}