#ifndef BALANCE_LB_DATA_TYPE_H
#define BALANCE_LB_DATA_TYPE_H

#include <unistd.h>

#include <string>
#include <memory>

enum LBTaskOp {
    LB_REGISTER_SERVER,
    LB_REGISTER_CACHE,
    LB_QUERY,
    LB_QUERY_COMPLETE,
    LB_TASK_REQUEST,
    LB_TASK_RESPONSE
};

struct LBTask {
    LBTaskOp op;
    uint64_t id;
    int server_ip;
    int server_port;
    std::string model_name;
    std::shared_ptr<char> data_ptr;
    size_t data_size;
};

#endif