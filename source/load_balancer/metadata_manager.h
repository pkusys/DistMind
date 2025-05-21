#ifndef BALANCE_LB_METADATA_MANAGER_H
#define BALANCE_LB_METADATA_MANAGER_H

#include <unistd.h>

#include <string>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

struct pair_hash {
    inline std::size_t operator()(const std::pair<int, int> & v) const {
        return (((uint64_t)v.first) << 31) + (uint64_t)v.second;
    }
};

class MetadataManager {
public:
    MetadataManager(std::string metadata_address, int metadata_port)
    :_metadata_address(metadata_address), _metadata_port(metadata_port) {};

    ~MetadataManager() {};

    size_t getModelSize(std::string model_name);
    size_t getCacheCapacity(int ip_int);
    size_t getCacheUsed(int ip_int);
    const std::unordered_set<std::string> getCacheSet(int ip_int);
    bool checkCache(int ip_int, std::string model_name);
    bool insertCache(int ip_int, std::string model_name);
    bool removeCache(int ip_int, std::string model_name);
    size_t getActivity(int ip_int, std::string model_name);

    void registerCache(int ip_int, size_t capacity);
    void registerServer(int ip_int, int port);

    void registerModel(int ip_int, int port, std::string model_name);
    void increaseWorkload(int ip_int, int port, std::string model_name);
    void decreaseWorkload(int ip_int, int port, std::string model_name);
    std::pair<int, int> getIdleServer(std::string model_name, int threshold);

private:
    std::string _metadata_address;
    int _metadata_port;

    std::unordered_map<std::string, size_t> _model_size; // Map: model_name -> size of model parameters
    std::unordered_map<int, size_t> _cache_capacity; // Map: ip_int -> allocated memory of caches
    std::unordered_map<int, size_t> _cache_used; // Map: ip_int -> used memory of caches
    std::unordered_map<int, std::unordered_set<std::string> > _cache_set; // Map: ip_int -> Set of cache on the instance
    std::unordered_map<int, std::unordered_map<std::string, int> > _cache_active; // Map: ip_int -> model_name -> pending requests

    std::pair<std::string, std::pair<int, int> > _flushed_model;
    std::unordered_map<std::string, std::unordered_set<std::pair<int, int>, pair_hash> > _model_location; // Map: model_name -> Set of (ip_int, port)
    std::unordered_map<int, std::unordered_map<int, std::string> > _location_model; // Map: ip_int -> (port -> model_name)
    std::unordered_map<int, std::unordered_map<int, int> > _workload_info; // Map: ip_int -> (port -> workload)
    std::mutex _lock;
};

#endif