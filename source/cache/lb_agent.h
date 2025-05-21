#ifndef BALANCE_CACHE_LB_AGENT_H
#define BALANCE_CACHE_LB_AGENT_H

#include <unistd.h>

#include <string>
#include <memory>

#include "utils/utils.h"

struct LBInstruction {
    int op; // 0 remove, 1 add model
    std::string key;

    LBInstruction(int _op, std::string _key):
    op(_op), key(_key) {}
};

class LBAgent {
public:
    LBAgent(std::string lb_addr, int lb_port, std::string addr_for_ser, size_t cache_capability);
    ~LBAgent();
    std::shared_ptr<LBInstruction> getInstruction(std::shared_ptr<balance::util::MemoryManager> _memory_manager);

private:
    std::shared_ptr<balance::util::TcpClient> _tcp;
};

#endif