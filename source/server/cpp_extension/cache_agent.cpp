#include <unistd.h>
#include <string.h>

#include <string>
#include <iostream>

#include <cuda_runtime.h>

#include "cache_agent.h"

using namespace std;
using namespace balance::util;

CacheAgent::CacheAgent(string addr, int port) {
    _client.reset(new TcpClient(addr, port));

    string shm_name;
    _client->tcpRecvString(shm_name);

    size_t shm_size = 0;
    _client->tcpRecv((char*)&shm_size, sizeof(shm_size));

    _shm.reset(new SharedMemory(shm_name, shm_size));
    cudaHostRegister((void*)_shm->getPointer(), _shm->getSize(), cudaHostRegisterDefault);
}

CacheAgent::~CacheAgent() {

}

char* CacheAgent::getShmPointer() {
    return _shm->getPointer();
}

size_t CacheAgent::getShmSize() {
    return _shm->getSize();
}

shared_ptr<BlockGroup> CacheAgent::get(string key) {
    _client->tcpSendString(key);
    size_t ret_length = 0;
    _client->tcpRecv((char*)&ret_length, sizeof(ret_length));
    shared_ptr<BlockGroup> ret((BlockGroup*)malloc(ret_length));
    _client->tcpRecv((char*)ret.get(), ret_length);
    return ret;
}