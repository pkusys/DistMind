#include <unistd.h>
#include <string.h>

#include <string>
#include <memory>
#include <thread>
#include <vector>
#include <iostream>
#include <unordered_map>

#include <cuda_runtime.h>

// #include "utils/utils.h"
#include "utils/common/atomic.h"
#include "utils/common/time.h"

#include "core_loops.h"
#include "cache_agent.h"
#include "operations.h"

using namespace std;
using namespace balance::util;

bool _shutdown;
vector<shared_ptr<thread> > _loops;
std::shared_ptr<CacheAgent> _cache_agent;
cudaStream_t _cuda_stream;

std::unordered_map<string, char*> _cached_info;
AtomicQueue<string> _queue_for_new_task;
AtomicQueue<pair<char*, size_t> > _queue_for_pyinfo;
AtomicQueue<string> _queue_for_param_recv;
AtomicQueue<shared_ptr<BlockGroup> > _queue_for_param_src;
AtomicQueue<shared_ptr<AddressPointer> > _queue_for_param_dst;
AtomicQueue<size_t> _queue_for_param_ready;

bool onceHandleTask() {
    if (!_queue_for_new_task.empty()) {
        string model_name = _queue_for_new_task.pop();

        bool cached = (_cached_info.find(model_name) != _cached_info.end());
        if (!cached) {
            shared_ptr<BlockGroup> ret = _cache_agent->get(model_name);
            if (ret->num > 1)
                cout << "Model Info may Overflow" << endl;

            _cached_info[model_name] = (char*)malloc(ret->size);
            char *dst = _cached_info[model_name];

            OffsetPointer* block_ptr = &(ret->data0);
            size_t accumulated_size = 0;
            for (size_t i = 0; i < ret->num; ++i) {
                char* ptr_src = _cache_agent->getShmPointer() + block_ptr[i].offset;
                char* ptr_dst = dst + accumulated_size;
                size_t size = block_ptr[i].size;
                memcpy(ptr_dst, ptr_src, size);
                accumulated_size += size;
            }
        }
        char* model_info = _cached_info[model_name];
        char* cursor = model_info;

        int pyinfo_length = *(int*)cursor; cursor += 4;
        char* pyinfo_cstr = (char*)cursor; cursor += pyinfo_length;
        cout << "Model Info: " << pyinfo_length << "Cached: " << cached << endl;
        pair<char*, size_t> pyinfo(pyinfo_cstr, pyinfo_length);
        if (!cached)
            _queue_for_pyinfo.push(pyinfo);
        int num_batch = *(int*)cursor; cursor += 4;
        for (int i = 0; i < num_batch; ++i) {
            int batch_id_length = *(int*)cursor; cursor += 4;
            char* batch_id_cstr = (char*)cursor; cursor += batch_id_length;
            string batch_id(batch_id_cstr, batch_id_length);
            // cout << "Batch ID: " << batch_id_length << "string: " << batch_id << endl;
            _queue_for_param_recv.push(batch_id);
        }
    }
    else {
        usleep(10);
    }
    return true;
}

bool onceRecv() {
    if (!_queue_for_param_recv.empty()) {
        string batch_id = _queue_for_param_recv.pop();
        // cout << "Waiting for batch ID: " << batch_id << endl;
        shared_ptr<BlockGroup> src = _cache_agent->get(batch_id);
        // cout << "Batch ID: " << batch_id << " size: " << src->size << endl;
        _queue_for_param_src.push(src);
    }
    else {
        usleep(10);
    }
    return true;
}

bool onceCopy() {
    if (!_queue_for_param_src.empty() && !_queue_for_param_dst.empty()) {
        char* shm_base = _cache_agent->getShmPointer();
        shared_ptr<BlockGroup> src = _queue_for_param_src.pop();
        shared_ptr<AddressPointer> dst = _queue_for_param_dst.pop();

        if (src->size != dst->size)
            cout << "Unmatched block size: " << src->size << ", " << dst->size << endl;

        OffsetPointer* block_ptr = &(src->data0);
        size_t accumulated_size = 0;
        for (size_t i = 0; i < src->num; ++i) {
            char* ptr_src = shm_base + block_ptr[i].offset;
            char* ptr_dst = dst->ptr + accumulated_size;
            size_t size = block_ptr[i].size;
            cudaMemcpyAsync(ptr_dst, ptr_src, size, cudaMemcpyHostToDevice, _cuda_stream);
            accumulated_size += size;
        }
        if(accumulated_size != src->size)
            cout << "Unmatched block copy: " << accumulated_size << ", " << src->size << endl;

        cudaStreamSynchronize(_cuda_stream);
        cout << "\tBatch in C++ " << fixed << time_now() << endl;
        _queue_for_param_ready.push(src->size);
        registerCopyback(src, shm_base, dst);
    }
    else {
        usleep(10);
    }
    return true;
}

void loopHandleTask() {
    while (onceHandleTask() && !_shutdown)
        continue;
}
void loopRecv() {
    while (onceRecv() && !_shutdown)
        continue;
}
void loopCopy() {
    while (onceCopy() && !_shutdown)
        continue;
}

void createLoops(string cache_addr, int cache_port) {
    _cache_agent.reset(new CacheAgent(cache_addr, cache_port));

    cudaStreamCreateWithFlags(&_cuda_stream, cudaStreamNonBlocking);
    
    _shutdown = false;
    _loops.push_back(shared_ptr<thread>(new thread(loopHandleTask)));
    _loops.push_back(shared_ptr<thread>(new thread(loopRecv)));
    _loops.push_back(shared_ptr<thread>(new thread(loopCopy)));
}
void destroyLoops() {
    _shutdown = true;
    while (!_loops.empty()) {
        _loops.back()->join();
        _loops.pop_back();
    }

    cudaStreamDestroy(_cuda_stream);
}

void enqueueTask(std::string model_name) {
    _queue_for_new_task.push(model_name);
}

pair<char*, size_t> dequeueModelInfo() {
    while (_queue_for_pyinfo.empty())
        usleep(10);
    return _queue_for_pyinfo.pop();
}

void enqueueParamGpuMemory(char* d_ptr, size_t size) {
    shared_ptr<AddressPointer> dst(new AddressPointer(d_ptr, size));
    _queue_for_param_dst.push(dst);
}

size_t dequeueParamCompletion() {
    while (_queue_for_param_ready.empty())
        usleep(10);
    return _queue_for_param_ready.pop();
}