#include <unistd.h>

#include <string>

#include <torch/torch.h>
#include <torch/extension.h>

#include "interface.h"
#include "../operations.h"

void torchInitializeServer(
    std::string addr_for_client, int port_for_client, 
    std::string cache_addr, int cache_port,
    std::string lb_addr, int lb_port
) {
    initializeServer(
        addr_for_client, port_for_client, 
        cache_addr, cache_port, 
        lb_addr, lb_port
    );
}

void torchFinalizeServer() {
    finalizeServer();
}

std::string torchGetTask() {
    return getTask();
}

pybind11::bytes torchGetData() {
    std::pair<std::shared_ptr<char>, size_t> data_cpp = getData();
    py::bytes data_b(data_cpp.first.get(), data_cpp.second);
    return data_b;
}

pybind11::bytes torchGetModelInfo() {
    std::pair<char*, size_t> pyinfo = getModelInfo();
    return pybind11::bytes(pyinfo.first, pyinfo.second);
}

void torchRegisterParamGpuMemory(torch::Tensor batch_start, size_t size) {
    registerParamGpuMemory((char*)batch_start.data_ptr(), size);
}

size_t torchCheckParamCompletion() {
    return checkParamCompletion();
}

void torchCopyback(bool train) {
    performCopyback(train);
}

void torchCompleteTask(std::string model_name, char* data, size_t size) {
    completeTask(model_name, data, size);
}

PYBIND11_MODULE(server_torch_c, m) {
    m.def("init_server", &torchInitializeServer, "torchInitializeServer");
    m.def("fin_server", &torchFinalizeServer, "torchFinalizeServer");
    m.def("get_task", &torchGetTask, "torchGetTask");
    m.def("get_data", &torchGetData, "torchGetData");
    m.def("get_model_info", &torchGetModelInfo, "torchGetModelInfo");
    m.def("register_param_gpu_memory", &torchRegisterParamGpuMemory, "torchRegisterParamGpuMemory");
    m.def("check_param_completion", &torchCheckParamCompletion, "torchCheckParamCompletion");
    m.def("copyback", &torchCopyback, "torchCopyback");
    m.def("complete_task", &torchCompleteTask, "torchCompleteTask");
}