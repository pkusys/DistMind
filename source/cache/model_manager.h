#ifndef BALANCE_CACHE_MODEL_MANAGER_H
#define BALANCE_CACHE_MODEL_MANAGER_H

#include <unistd.h>

#include <string>
#include <memory>
#include <unordered_map>
#include <mutex>

#include "utils/utils.h"

class ModelManager {
public:
    ModelManager(std::shared_ptr<balance::util::MemoryManager> deallocator);
    ~ModelManager();
    std::shared_ptr<balance::util::BlockGroup> get(std::string key);
    void put(std::string key, std::shared_ptr<balance::util::BlockGroup> item);
    void erase(std::string key);

private:
    std::shared_ptr<balance::util::MemoryManager> _deallocator;
    std::unordered_map<std::string, std::shared_ptr<balance::util::BlockGroup> > _data;
    std::mutex _lock;
};

#endif