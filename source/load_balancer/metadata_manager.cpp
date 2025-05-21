#include <unistd.h>

#include <string>
#include <iostream>
#include <memory>

#include "metadata_manager.h"
#include "utils/utils.h"

using namespace std;
using namespace balance::util;

void MetadataManager::registerServer(int ip_int, int port) {
    const lock_guard<mutex> guard(_lock);
    if (_workload_info.find(ip_int) == _workload_info.end())
        _workload_info[ip_int] = unordered_map<int, int>();
    if (_workload_info[ip_int].find(port) == _workload_info[ip_int].end())
        _workload_info[ip_int][port] = 0;
}

void MetadataManager::registerCache(int ip_int, size_t capacity) {
    _cache_capacity[ip_int] = capacity;
    _cache_used[ip_int] = 0;
}

void MetadataManager::registerModel(int ip_int, int port, string model_name) {
    const lock_guard<mutex> guard(_lock);
    string current_model = _location_model[ip_int][port];
    size_t erased = _model_location[current_model].erase(make_pair(ip_int, port));
    _model_location[model_name].insert(make_pair(ip_int, port));
    _location_model[ip_int][port] = model_name;
    _cache_active[ip_int][current_model] -= 1;
    _cache_active[ip_int][model_name] += 1;

    if (erased > 0)
        _flushed_model = make_pair(current_model, make_pair(ip_int, port));
    else
        _flushed_model = make_pair("", make_pair(0, 0));
}

size_t MetadataManager::getModelSize(string model_name) {
    // const lock_guard<mutex> guard(_lock);
    size_t model_size = _model_size[model_name];
    if (model_size == 0) {
        shared_ptr<TcpClient> storage(new TcpClient(_metadata_address, _metadata_port));

        int op = KVSTORAGE_OP_READ;
        storage->tcpSend((char*)&op, sizeof(op));

        string metadata_key = model_name + string("-SIZE");
        storage->tcpSendString(metadata_key);

        string model_size_b;
        storage->tcpRecvString(model_size_b);
        model_size = *(size_t*)model_size_b.c_str();
        model_size *= MEMORY_MANAGER_AMPLIFIER;
        _model_size[model_name] = model_size;
    }
    return model_size;
}
size_t MetadataManager::getCacheCapacity(int ip_int) {
    // const lock_guard<mutex> guard(_lock);
    return _cache_capacity[ip_int];
}
size_t MetadataManager::getCacheUsed(int ip_int) {
    // const lock_guard<mutex> guard(_lock);
    return _cache_used[ip_int];
}
const unordered_set<string> MetadataManager::getCacheSet(int ip_int) {
    // const lock_guard<mutex> guard(_lock);
    return _cache_set[ip_int];
}
bool MetadataManager::checkCache(int ip_int, string model_name) {
    // const lock_guard<mutex> guard(_lock);
    return _cache_set[ip_int].find(model_name) != _cache_set[ip_int].end();
}
bool MetadataManager::insertCache(int ip_int, string model_name) {
    // const lock_guard<mutex> guard(_lock);
    if (checkCache(ip_int, model_name))
        return true;

    if (_cache_capacity[ip_int] - _cache_used[ip_int] < getModelSize(model_name))
        return false;

    _cache_set[ip_int].insert(model_name);
    _cache_used[ip_int] += getModelSize(model_name);
    return true;
}
bool MetadataManager::removeCache(int ip_int, string model_name) {
    // const lock_guard<mutex> guard(_lock);
    if (_cache_set[ip_int].find(model_name) != _cache_set[ip_int].end()) {
        _cache_set[ip_int].erase(model_name);
        _cache_used[ip_int] -= getModelSize(model_name);
        return true;
    }
    return false;
}
size_t MetadataManager::getActivity(int ip_int, string model_name) {
    const lock_guard<mutex> guard(_lock);
    if (_cache_active[ip_int].find(model_name) == _cache_active[ip_int].end())
        return 0;
    return _cache_active[ip_int][model_name];
}

void MetadataManager::increaseWorkload(int ip_int, int port, string model_name) {
    const lock_guard<mutex> guard(_lock);
    _workload_info[ip_int][port]++;
    _cache_active[ip_int][model_name] += 1;
    // cout << "Cache Liveness" << ", " << (unsigned int)ip_int << ", " << model_name << ", " << _cache_active[ip_int][model_name] << endl;
}
void MetadataManager::decreaseWorkload(int ip_int, int port, string model_name) {
    const lock_guard<mutex> guard(_lock);
    _workload_info[ip_int][port]--;
    _cache_active[ip_int][model_name] -= 1;
    // cout << "Cache Liveness" << ", " << (unsigned int)ip_int << ", " << model_name << ", " << _cache_active[ip_int][model_name] << endl;
}

pair<int, int> MetadataManager::getIdleServer(string model_name, int threshold) {
    const lock_guard<mutex> guard(_lock);
    pair<int, int> idle_server(0, 0);
    if (model_name.compare(_flushed_model.first) == 0)
        idle_server = _flushed_model.second;

    int minimum_workload = threshold;
    for (auto itr = _model_location[model_name].begin(); itr != _model_location[model_name].end(); ++itr) {
        int ip_int = itr->first;
        int port = itr->second;
        int workload = _workload_info[ip_int][port];
        if (workload < minimum_workload) {
            idle_server = *itr;
            minimum_workload = workload;
        }
    }
    return idle_server;
}