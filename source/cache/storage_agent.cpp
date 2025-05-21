#include <unistd.h>
#include <arpa/inet.h>
#include <string.h>

#include <string>

#include "storage_agent.h"

using namespace std;
using namespace balance::util;
using namespace pipeps::store;

StorageConnection::StorageConnection(uint64_t comm_id, string shm_name, size_t shm_size, int num_worker):
StoreCli(to_string(comm_id), commidToIp(comm_id), to_string(commidToPort(comm_id)), shm_name, shm_size, num_worker) {
    _counter = getCommCntr();
    cout << "Connect to " << commidToIp(comm_id) << ":" << commidToPort(comm_id) << endl;
}

StorageConnection::~StorageConnection() {}

uint64_t StorageConnection::connRecvAsync(string key, vector<OffsetPointer> tasks, uint64_t id) {
    if (id == 0)
        id = getNextID();
    
    vector<pair<size_t, size_t> > dataLoc;
    for (auto itr = tasks.begin(); itr != tasks.end(); ++itr)
        dataLoc.push_back(make_pair<>(itr->offset, itr->size));
    pull(key, dataLoc);
    // cout << string("StorageConnection::connRecvAsync :: size of dataLoc" + to_string(dataLoc.size()) + "\n");
    _pending_tasks_queue.push(make_pair<>(id, tasks.size()));
    _pending_tasks_set.insert(id);
    return id;
}

bool StorageConnection::connCheckCompletion(uint64_t id) {
    int _next_counter = getCommCntr();
    while ((!_pending_tasks_queue.empty()) && _counter + _pending_tasks_queue.front().second <= _next_counter) {
        _pending_tasks_set.erase(_pending_tasks_queue.front().first);
        _counter += _pending_tasks_queue.front().second;
        _pending_tasks_queue.pop();
    }

    return _pending_tasks_set.find(id) == _pending_tasks_set.end();
}

uint64_t StorageConnection::getNextID() {
    static uint64_t id_generator = 0;
    return ++id_generator;
}

StorageAgent::StorageAgent(string addr, int port, shared_ptr <MemoryManager> allocator):
_allocator(allocator), _metadata_storage_addr(addr), _metadata_storage_port(port) {
    // _metadata_storage.reset(new TcpClient(addr, port));
}

StorageAgent::~StorageAgent() {}

shared_ptr<BlockGroup> StorageAgent::getAsync(string key) {
    const lock_guard<mutex> guard(_lock);

    if (_kv_slice_location.find(key) == _kv_slice_location.end()) {
        auto metadata_v = getSliceLocation(key);

        _kv_slice_location[key] = metadata_v;

        for (auto itr = metadata_v.begin(); itr != metadata_v.end(); ++itr) {
            if (_connections.find(itr->comm_id) == _connections.end())
                _connections[itr->comm_id] = shared_ptr<StorageConnection>(new StorageConnection(itr->comm_id, _allocator->getShmName(), _allocator->getShmSize()));
        }

        size_t total_size = 0;
        for (auto itr = metadata_v.begin(); itr != metadata_v.end(); ++itr)
            total_size += itr->size;
        _kv_size[key] = total_size;
    }

    // cout << key << ", " << _kv_size[key] << ", " << _allocator->getBlockAvailable() << endl;
    shared_ptr<BlockGroup> bg = _allocator->allocate(_kv_size[key]);
    vector<KVSliceMetadata> kv_slices = _kv_slice_location[key];
    vector<shared_ptr<StorageConnection> > used_conn;
    for (auto slice = kv_slices.begin(); slice != kv_slices.end(); ++slice) {
        shared_ptr<StorageConnection> conn = _connections[slice->comm_id];

        string slice_key(key
            + "-SLICE-OFFSET-" + to_string(slice->offset)
            + "-SLICE-SIZE-" + to_string(slice->size)
        );
        
        vector<OffsetPointer> fractions = bg->accessFraction(slice->offset, slice->size);

        conn->connRecvAsync(slice_key, fractions, (uint64_t)bg.get());
        
        if (!conn->connCheckCompletion((uint64_t)bg.get()))
            used_conn.push_back(conn);
    }

    _pending_query[bg] = used_conn;
    return bg;
}

bool StorageAgent::checkGetAsync(shared_ptr<BlockGroup> bg) {
    const lock_guard<mutex> guard(_lock);

    auto itr = _pending_query.find(bg);
    if (itr == _pending_query.end())
        return true;

    for (shared_ptr<StorageConnection> conn: itr->second) {
        if (!conn->connCheckCompletion((uint64_t)bg.get()))
            return false;
    }
    _pending_query.erase(itr);
    return true;
}

vector<KVSliceMetadata> StorageAgent::getSliceLocation(string key) {    
    shared_ptr<TcpClient> _metadata_storage(new TcpClient(_metadata_storage_addr, _metadata_storage_port));

    _metadata_storage->tcpSend((char*)&KVSTORAGE_OP_READ, sizeof(KVSTORAGE_OP_READ));

    string location_key = key + string("-LOCATION");
    _metadata_storage->tcpSendString(location_key);

    string location;
    _metadata_storage->tcpRecvString(location);
    
    int metadata_num = location.size() / sizeof(KVSliceMetadata);
    vector<KVSliceMetadata> metadata_v((KVSliceMetadata*)location.c_str(), (KVSliceMetadata*)location.c_str() + metadata_num);
    return metadata_v;
}