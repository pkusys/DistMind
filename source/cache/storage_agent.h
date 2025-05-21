#ifndef BALANCE_CACHE_STORAGE_AGENT_H
#define BALANCE_CACHE_STORAGE_AGENT_H

#include <unistd.h>
#include <arpa/inet.h>

#include <string>
#include <unordered_map>
#include <mutex>
#include <queue>
#include <unordered_set>

#include "utils/utils.h"

const size_t kMaxSmallData = 128 * 1024;

struct KVSliceMetadata {
    size_t offset;
    size_t size;
    uint64_t comm_id;
};

class StorageConnection: public pipeps::store::StoreCli {
public:
    StorageConnection(uint64_t comm_id, std::string shm_name, size_t shm_size, int num_worker = 4);
    ~StorageConnection();

    uint64_t connRecvAsync(std::string key, std::vector<balance::util::OffsetPointer> tasks, uint64_t id = 0);
    bool connCheckCompletion(uint64_t id);

private:
    static uint64_t getNextID();

private:
    int _counter;
    std::queue<std::pair<uint64_t, int> > _pending_tasks_queue;
    std::unordered_set<uint64_t> _pending_tasks_set;
};

class StorageAgent {
public:
    StorageAgent(std::string addr, int port, std::shared_ptr<balance::util::MemoryManager> allocator);
    ~StorageAgent();
    std::shared_ptr<balance::util::BlockGroup> getAsync(std::string key);
    bool checkGetAsync(std::shared_ptr<balance::util::BlockGroup> bg);

private:
    std::vector<KVSliceMetadata> getSliceLocation(std::string key);

private:
    std::shared_ptr<balance::util::MemoryManager> _allocator;
    // std::shared_ptr<balance::util::TcpClient> _metadata_storage;
    std::string _metadata_storage_addr;
    int _metadata_storage_port;
    std::unordered_map<uint64_t, std::shared_ptr<StorageConnection> > _connections;
    std::unordered_map<std::string, std::vector<KVSliceMetadata> > _kv_slice_location;
    std::unordered_map<std::string, size_t> _kv_size;
    
    std::unordered_map<std::shared_ptr<balance::util::BlockGroup>, std::vector<std::shared_ptr<StorageConnection> > > _pending_query;
    std::mutex _lock;
};

inline std::string commidToIp(uint64_t comm_id) {
    char ip_addr[INET_ADDRSTRLEN];
    memset(ip_addr, 0, INET_ADDRSTRLEN);
    inet_ntop(AF_INET, (int*)&comm_id, ip_addr, INET_ADDRSTRLEN);
    return std::string(ip_addr);
}

inline int commidToPort(uint64_t comm_id) {
    return *((int*)(&comm_id) + 1);
}

#endif