#include <unistd.h>

#include <string>
#include <iostream>

#include "metadata_manager.h"

using namespace std;
using namespace balance::util;

void MetadataManager::register_cache(int ip_int, size_t capacity) {
    _cache_limit[ip_int] = capacity;
    _cache_used[ip_int] = 0;
}

void MetadataManager::register_server(int ip_int, int port) {
    if (_workload_info.find(ip_int) == _workload_info.end())
        _workload_info[ip_int] = unordered_map<int, int>();
    if (_workload_info[ip_int].find(port) == _workload_info[ip_int].end())
        _workload_info[ip_int][port] = 0;
}

const unordered_map<string, unordered_set<int> >& MetadataManager::get_cache_location() {
    return _cache_location;
}
const unordered_map<int, unordered_map<int, int> >& MetadataManager::get_workload() {
    return _workload_info;
}

size_t MetadataManager::get_cache_limit(int ip_int) {
    return _cache_limit[ip_int];
}

size_t MetadataManager::get_cache_used(int ip_int) {
    return _cache_used[ip_int];
}

size_t MetadataManager::get_model_size(std::string model_name) {
    auto itr = _model_size.find(model_name);
    if (itr == _model_size.end())
        return 0;
    else
        return itr->second;
}

void MetadataManager::set_model_size(std::string model_name, size_t size) {
    _model_size[model_name] = size;
}

int MetadataManager::check_cache(std::string model_name) {
    if (_cache_location.find(model_name) == _cache_location.end())
        return 0;
    if (_cache_location[model_name].empty())
        return 0;
    return *(_cache_location[model_name].begin());
}

int MetadataManager::check_cache_space(size_t size) {
    int selection_ip_int = 0;
    size_t max_cache_space = size - 1;
    for (auto itr = _cache_limit.begin(); itr != _cache_limit.end(); ++itr) {
        size_t limit = itr->second;
        size_t used = _cache_used[itr->first];
        size_t available = limit - used;
        if (available > max_cache_space) {
            selection_ip_int = itr->first;
            max_cache_space = available;
        }
    }
    cout << "Available cache: " << selection_ip_int << ", " << max_cache_space << endl;
    return selection_ip_int;
}

void MetadataManager::cache_in_model(int ip_int, string model_name) {
    if (_cache_location[model_name].find(ip_int) == _cache_location[model_name].end())
        _cache_used[ip_int] += get_model_size(model_name) * MEMORY_MANAGER_AMPLIFIER;
    // _lru_cache_all.push(model_name);
    if (_model_freq.find(model_name) == _model_freq.end()) {
        _model_freq[model_name] = 0;
        _model_ref[model_name] = 0;
    }
    _model_freq[model_name] += 1;
    _model_ref[model_name] += 1;
    _cache_location[model_name].insert(ip_int);
    _model_cached.insert(model_name);
}

pair<int, string> MetadataManager::cache_out_model() {
    // string model_name = _lru_cache_all.pop();
    string model_name("");
    size_t min_freq = -1;
    for (auto itr = _model_cached.begin(); itr != _model_cached.end(); ++itr) {
        if (_model_ref[*itr] > 0)
            continue;
        if (_model_freq[*itr] < min_freq) {
            min_freq = _model_freq[*itr];
            model_name = *itr;
        }
    }

    int ip_int = 0;
    if (model_name.length() > 0) {
        ip_int = *(_cache_location[model_name].begin());
        _cache_location[model_name].erase(ip_int);
        _cache_used[ip_int] -= get_model_size(model_name) * MEMORY_MANAGER_AMPLIFIER;
        _model_cached.erase(model_name);
    }
    return make_pair<>(ip_int, model_name);
}

void MetadataManager::increase_workload(int ip_int, int port, string model_name) {
    _workload_info[ip_int][port]++;
}
void MetadataManager::decrease_workload(int ip_int, int port, string model_name) {
    _workload_info[ip_int][port]--;
    // _lru_cache_all.finish(model_name);
    _model_ref[model_name] -= 1;
}
int MetadataManager::check_idle_server() {
    int max_idle = -1024;
    for (auto itr_machine = _workload_info.begin(); itr_machine != _workload_info.end(); ++itr_machine) {
        auto machine_info = itr_machine->second;
        int idle_index = machine_info.size();
        for (auto itr_gpu = machine_info.begin(); itr_gpu != machine_info.end(); ++itr_gpu) {
            idle_index -= itr_gpu->second;
        }
        if (idle_index > max_idle)
            max_idle = idle_index;
    }
    return max_idle;
}