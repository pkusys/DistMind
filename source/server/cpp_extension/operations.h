#ifndef BALANCE_SERVER_OPERATION_H
#define BALANCE_SERVER_OPERATION_H

#include <unistd.h>

#include <string>

#include "utils/common/pointer.h"
#include "utils/memory_manager/memory_manager.h"

void initializeServer(
    std::string addr_for_client, int port_for_client, 
    std::string cache_addr, int cache_port,
    std::string lb_addr, int lb_port
);
void finalizeServer();
std::string getTask();
std::pair<std::shared_ptr<char>, size_t> getData();
std::pair<char*, size_t> getModelInfo();
void registerParamGpuMemory(char* d_ptr, size_t size);
size_t checkParamCompletion();
void registerCopyback(std::shared_ptr<balance::util::BlockGroup> cpu, char* shm_base, std::shared_ptr<balance::util::AddressPointer> gpu);
void performCopyback(bool train);
void completeTask(std::string model_name, char* data, size_t size);

#endif