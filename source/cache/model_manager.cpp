#include <unistd.h>

#include <string>

#include "model_manager.h"

using namespace std;
using namespace balance::util;

ModelManager::ModelManager(shared_ptr<balance::util::MemoryManager> deallocator):
_deallocator(deallocator) {}

ModelManager::~ModelManager() {}

shared_ptr<BlockGroup> ModelManager::get(std::string key) {
    const lock_guard<mutex> guard(_lock);
    auto itr = _data.find(key);
    if (itr != _data.end())
        return itr->second;
    else
        return nullptr;
}

void ModelManager::put(std::string key, shared_ptr<BlockGroup> item) {
    const lock_guard<mutex> guard(_lock);
    _data[key] = item;
}

void ModelManager::erase(std::string key) {
    const lock_guard<mutex> guard(_lock);
    auto itr = _data.find(key);
    if (itr != _data.end()) {
        _deallocator->deallocate(itr->second);
        _data.erase(itr);
    }
}