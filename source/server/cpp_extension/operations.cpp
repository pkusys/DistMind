#include <unistd.h>

#include <string>
#include <memory>
#include <queue>

#include <cuda_runtime.h>

#include "operations.h"
#include "lb_agent.h"
#include "core_loops.h"

using namespace std;
using namespace balance::util;

shared_ptr<LBAgent> _lb_agent;
queue<pair<pair<shared_ptr<BlockGroup>, char*>, shared_ptr<AddressPointer> > > _queue_for_copyback;
string current_model;

void initializeServer(
    string addr_for_client, int port_for_client, 
    string cache_addr, int cache_port,
    string lb_addr, int lb_port
) {
    _lb_agent.reset(new LBAgent(lb_addr, lb_port, addr_for_client, port_for_client));
    createLoops(cache_addr, cache_port);
}

void finalizeServer() {
    destroyLoops();
}

string getTask() {
    // (Enqueue) Send model name as task to loops
    string model_name = _lb_agent->getTask();
    if (current_model != model_name) {
        current_model = model_name;
        enqueueTask(model_name);
    }
    return model_name;
}

pair<shared_ptr<char>, size_t> getData() {
    return _lb_agent->getData();
}

pair<char*, size_t> getModelInfo() {
    // (Dequeue) Return model info to Python
    return dequeueModelInfo();
}

void registerParamGpuMemory(char* d_ptr, size_t size) {
    // (Enqueue) Send GPU pointer to workers for transmission
    enqueueParamGpuMemory(d_ptr, size);
}

size_t checkParamCompletion() {
    // (Dequeue) Check whether there is a ready batch
    return dequeueParamCompletion();
}

void registerCopyback(shared_ptr<BlockGroup> cpu, char* shm_base, shared_ptr<AddressPointer> gpu) {
    _queue_for_copyback.push(make_pair<>(make_pair<>(cpu, shm_base), gpu));
}

void performCopyback(bool train) {
    if (train) {
        while (! _queue_for_copyback.empty()) {            
            auto cpu = _queue_for_copyback.front().first.first;
            char* shm_base = _queue_for_copyback.front().first.second;
            auto gpu = _queue_for_copyback.front().second;
            _queue_for_copyback.pop();

            OffsetPointer* block_ptr = &(cpu->data0);
            size_t accumulated_size = 0;
            for (size_t i = 0; i < cpu->num; ++i) {
                char* ptr_src = gpu->ptr + accumulated_size;
                char* ptr_dst = shm_base + block_ptr[i].offset;
                size_t size = block_ptr[i].size;
                cudaMemcpy(ptr_dst, ptr_src, size, cudaMemcpyDeviceToHost);
                accumulated_size += size;
            }
        }
    }
    else {
        queue<pair<pair<shared_ptr<BlockGroup>, char*>, shared_ptr<AddressPointer> > > empty;
        swap(_queue_for_copyback, empty);
    }
}

void completeTask(string model_name, char* data, size_t size) {
    _lb_agent->completeTask(model_name, data, size);
}