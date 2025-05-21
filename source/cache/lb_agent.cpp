#include <unistd.h>
#include <string.h>
#include <arpa/inet.h>

#include <string>

#include "lb_agent.h"

using namespace std;
using namespace balance::util;

LBAgent::LBAgent(std::string lb_addr, int lb_port, std::string addr_for_ser, size_t capability) {
    _tcp.reset(new TcpClient(lb_addr, lb_port));

    struct {
        int ip_int;
        int port;
        size_t capacity;
    } cache_info;
    inet_pton(AF_INET, addr_for_ser.c_str(), &(cache_info.ip_int));
    cache_info.port = 0;
    cache_info.capacity = capability;
    _tcp->tcpSend((char*)&cache_info, sizeof(cache_info));
    cout << "Connect to the controller: " << (unsigned int)cache_info.ip_int << ", " << cache_info.capacity << endl << endl;
}

LBAgent::~LBAgent() {

}

shared_ptr<LBInstruction> LBAgent::getInstruction(shared_ptr<MemoryManager> _memory_manager) {
    int op;
    _tcp->tcpRecv((char*)&op, sizeof(op));

    // size_t available_memory = _memory_manager->getBlockAvailable() * _memory_manager->getBlockSize();
    // _tcp->tcpSend((char*)&available_memory, sizeof(available_memory));
    
    string model_name;
    _tcp->tcpRecvString(model_name);
    
    int reply = 1;
    _tcp->tcpSend((char*)&reply, sizeof(reply));

    // cout << "Get instruction: " << available_memory << ", " << op << ", " << model_name << endl;

    if (model_name.length() > 0)
        return shared_ptr<LBInstruction>(new LBInstruction(op, model_name));
    return nullptr;
}