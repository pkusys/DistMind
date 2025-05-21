#ifndef BALANCE_SERVER_TORCH_INTERFACE_H
#define BALANCE_SERVER_TORCH_INTERFACE_H

#include <unistd.h>

#include <string>

#include <pybind11/pybind11.h>
#include <torch/torch.h>
#include <torch/extension.h>

void torchInitializeServer(
    std::string addr_for_client, int port_for_client, 
    std::string cache_addr, int cache_port,
    std::string lb_addr, int lb_port
);
void torchFinalizeServer();
std::string torchGetTask();
pybind11::bytes torchGetData();
pybind11::bytes torchGetModelInfo();
void torchRegisterParamGpuMemory(torch::Tensor batch_start, size_t size);
size_t torchCheckParamCompletion();
void torchCopyback(bool train);
void torchCompleteTask(std::string model_name, char* data, size_t size);

#endif