#include <unistd.h>
#include <string.h>

#include <string>
#include <memory>
#include <iostream>
#include <thread>

#include "utils/utils.h"

#include "model_manager.h"
#include "storage_agent.h"
#include "model_manager.h"
#include "lb_agent.h"
#include <vector>
using namespace std;
using namespace balance::util;

enum ManageOp {
    ERASE_MODEL,
    FETCH_MODEL,
    FETCH_BATCH,
    MODEL_READY,
    BATCH_READY
};

struct ServerQuery {
    shared_ptr<TcpAgent> agent;
    std::string key;

    ServerQuery(shared_ptr<TcpAgent> _agent, std::string _key):
    agent(_agent), key(_key) {}
};

struct ManageInstruction {
    ManageOp op;
    string key;
    shared_ptr<BlockGroup> data;

    ManageInstruction(ManageOp _op, string _key, shared_ptr<BlockGroup> _data):
    op(_op), key(_key), data(_data) {}

    ManageInstruction(const ManageInstruction &mi):
    op(mi.op), key(mi.key), data(mi.data) {}
};

struct AgentHandlerForServer: public AgentHandler {
    void run(shared_ptr<TcpAgent> agent);
};

shared_ptr<MemoryManager> _memory_manager;
shared_ptr<StorageAgent> _storage_agent;
shared_ptr<ModelManager> _model_manager;
shared_ptr<LBAgent> _lb_agent;

bool _shutdown;
AtomicQueue<shared_ptr<ServerQuery> > _queue_for_query;
AtomicQueue<shared_ptr<ManageInstruction> > _queue_for_fetch;
AtomicQueue<shared_ptr<ManageInstruction> > _queue_for_manage;

bool onceHandleQuery();
bool onceManage();
bool onceForStorage();
bool onceForLB();
void loopHandleQuery();
void loopManage();
void loopForStorage();
void loopForLB();

vector<pair<string, int *>> model_infos;
void printModelInfo() {
    cout << time_now() << " Model Info: " << endl;
    for (auto &model_info : model_infos) {
        cout << "Model Name: " << model_info.first << ", Size: " << *(model_info.second) << endl;
    }
}

inline string getManageOpMsg(ManageOp op) {
    switch (op)
    {
    case ERASE_MODEL:
        return "ERASE_MODEL";
    case FETCH_MODEL:
        return "FETCH_MODEL";
    case FETCH_BATCH:
        return "FETCH_BATCH";
    case MODEL_READY:
        return "MODEL_READY";
    case BATCH_READY:
        return "BATCH_READY";
    default:
        return "";
    }
}

int main(int argc, char** argv) {
    if (argc < 8) {
        cout << string("Argument Error") << endl;
        cout << string("program [AddrForServer] [PortForServer] [MEtaStoreAddr] [MetaStorePort] [LBAddr] [LBPort] [ShmName] [ShmSize] [ShmBlockSize]") << endl;
    }

    string address_for_server(argv[1]);
    int port_for_server = stoi(argv[2]);
    string meta_store_address(argv[3]);
    int meta_store_port = stoi(argv[4]);
    string lb_address(argv[5]);
    int lb_port = stoi(argv[6]);
    string shm_name(argv[7]);
    size_t shm_size = stoull(argv[8]);
    size_t shm_block_size = stoull(argv[9]);

    // Connect to storage
    _memory_manager.reset(new MemoryManager(shm_name, shm_size, shm_block_size));
    _storage_agent.reset(new StorageAgent(meta_store_address, meta_store_port, _memory_manager));
    
    // Initialize data manager
    _model_manager.reset(new ModelManager(_memory_manager));
    
    // Connect to load balancer
    _lb_agent.reset(new LBAgent(lb_address, lb_port, address_for_server, shm_size));

    // Create loops
    thread t_query(loopHandleQuery);
    thread t_manage(loopManage);
    thread t_storage(loopForStorage);
    thread t_lb(loopForLB);
    
    // Listen to servers
    cout << "Ready to accept servers" << endl;
    TcpServerParallelWithAgentHandler s_server(address_for_server, port_for_server, shared_ptr<AgentHandler>(new AgentHandlerForServer()));

    t_query.join();
    t_manage.join();
    t_storage.join();
    t_lb.join();
    return 0;
}

void AgentHandlerForServer::run(shared_ptr<TcpAgent> agent) {
    string shm_name = _memory_manager->getShmName();
    size_t shm_size = _memory_manager->getShmSize();
    agent->tcpSendString(shm_name);
    agent->tcpSend((char*)&shm_size, sizeof(shm_size));

    while (true) {
        // Get Key
        string key;
        agent->tcpRecvString(key);

        // Enqueue
        shared_ptr<ServerQuery> query(new ServerQuery(agent, key));
        _queue_for_query.push(query);
    }
}

bool onceHandleQuery() {
    if (!_queue_for_query.empty()) {
        cout << "Query: " << fixed << time_now() << endl;
        // printModelInfo();
        shared_ptr<ServerQuery> query = _queue_for_query.pop();
        shared_ptr<BlockGroup> data = _model_manager->get(query->key);
        if (data != nullptr) {
            cout << "Get key data" << endl;
            cout << "Key: " << query->key << endl;
            size_t msg_size = sizeof(BlockGroup) + sizeof(OffsetPointer) * (data->num - 1);
            cout << "Msg Size: " << msg_size << endl;
            query->agent->tcpSend((char*)&msg_size, sizeof(msg_size));
            query->agent->tcpSend((char*)data.get(), msg_size);
            char * cursor = _memory_manager->getShmPointer() + data->data0.offset;
            int length = *(int*)cursor;
            cout << "Model Info: " << length << endl;
        }
        else {
            _queue_for_query.push(query);
        }
    }
    else {
        usleep(10);
    }

    return true;
}

bool onceManage() {
    if (!_queue_for_manage.empty()) {
        shared_ptr<ManageInstruction> instr = _queue_for_manage.pop();
        cerr << fixed << time_now() << " Get instruction: " << getManageOpMsg(instr->op) << ", " << instr->key << endl;
        // printModelInfo();
        if (instr->op == FETCH_MODEL) {
            string key = instr->key + "-METADATA";
            instr->data = _storage_agent->getAsync(key);
            _queue_for_fetch.push(instr);
        }
        else if (instr->op == MODEL_READY) {
            // cerr << fixed << time_now() << " Model ready: " << instr->key << endl;
            _model_manager->put(instr->key, instr->data);
            char* model_info = nullptr;
            if(instr->data->num > 1)
            {
                cout << "Model info may overflow :" << instr->data->size << endl;
                char *dst = (char *)malloc(instr->data->size);
                OffsetPointer* block_ptr = &(instr->data->data0);
                size_t accumulated_size = 0;
                for (size_t i = 0; i < instr->data->num; ++i) {
                    char* ptr_src = _memory_manager->getShmPointer() + block_ptr[i].offset;
                    char* ptr_dst = dst + accumulated_size;
                    size_t size = block_ptr[i].size;
                    memcpy(ptr_dst, ptr_src, size);
                    accumulated_size += size;
                }
                model_info = dst;
            }
            else
                model_info = _memory_manager->getShmPointer() + instr->data->data0.offset;

            char* cursor = (char*)model_info;
            int pyinfo_length = *(int*)cursor;
            cout << "Model Info: " << pyinfo_length << endl;
            model_infos.push_back(make_pair(instr->key, (int *)(_memory_manager->getShmPointer() + instr->data->data0.offset)));
            cursor += (4 + pyinfo_length);
            int num_batch = *(int*)cursor; cursor += 4;
            for (int i = 0; i < num_batch; ++i) {
                int batch_id_length = *(int*)cursor; cursor += 4;
                char* batch_id_cstr = (char*)cursor; cursor += batch_id_length;
                string batch_id(batch_id_cstr, batch_id_length);
                cout << "Batch ID: " << batch_id_length << "string: " << batch_id << endl;
                _queue_for_fetch.push(shared_ptr<ManageInstruction>(new ManageInstruction(FETCH_BATCH, batch_id, _storage_agent->getAsync(batch_id))));
            }
            if(instr->data->num > 1)
                free(model_info);
        }
        else if (instr->op == ERASE_MODEL) {
            cerr << fixed << time_now() << " Erase Model: " << instr->key << endl;
            shared_ptr<BlockGroup> data = _model_manager->get(instr->key);
            char* model_info = nullptr;
            if(data->num > 1)
            {
                cout << "Model info may overflow" << endl;
                char *dst = (char *)malloc(instr->data->size);
                OffsetPointer* block_ptr = &(instr->data->data0);
                size_t accumulated_size = 0;
                for (size_t i = 0; i < instr->data->num; ++i) {
                    char* ptr_src = _memory_manager->getShmPointer() + block_ptr[i].offset;
                    char* ptr_dst = dst + accumulated_size;
                    size_t size = block_ptr[i].size;
                    memcpy(ptr_dst, ptr_src, size);
                    accumulated_size += size;
                }
                model_info = dst;
            }
            else
                model_info = _memory_manager->getShmPointer() + data->data0.offset;
            char* cursor = (char*)model_info;
            int pyinfo_length = *(int*)cursor;
            cursor += (4 + pyinfo_length);
            int num_batch = *(int*)cursor; cursor += 4;
            for (int i = 0; i < num_batch; ++i) {
                int batch_id_length = *(int*)cursor; cursor += 4;
                char* batch_id_cstr = (char*)cursor; cursor += batch_id_length;
                string batch_id(batch_id_cstr, batch_id_length);
                _model_manager->erase(batch_id);
            }
            if(data->num > 1)
                free(model_info);
            _model_manager->erase(instr->key);
        }
        else if (instr->op == BATCH_READY) {
            cerr << fixed << time_now() << " Batch ready: " << instr->key << endl;
            _model_manager->put(instr->key, instr->data);
        }
        else {
            cerr << "Op Error In loopManage: " << instr->op << endl;
        }
        cerr << "Instruction complete" << endl << endl;
        // printModelInfo();
    }
    else {
        usleep(10);
    }
    return true;
}

bool onceForStorage() {
    if (!_queue_for_fetch.empty()) {
        shared_ptr<ManageInstruction> instr = _queue_for_fetch.pop();
        // if (instr->data == nullptr) {
        //     string key = instr->op == FETCH_MODEL? (instr->key + "-METADATA"): instr->key;
        //     instr->data = _storage_agent->getAsync(key);
        //     // cerr << fixed << time_now() << " Fetch: " << instr->key << endl;
        // }
        if (_storage_agent->checkGetAsync(instr->data)) {
            ManageOp op = instr->op == FETCH_MODEL? MODEL_READY: BATCH_READY;
            _queue_for_manage.push(shared_ptr<ManageInstruction>(new ManageInstruction(op, instr->key, instr->data)));
            // cerr << fixed << time_now() << " Fetch complete: " << instr->key << endl;
        }
        else {
            _queue_for_fetch.push(instr);
        }
    }
    else {
        usleep(10);
    }

    return true;
}

bool onceForLB() {
    shared_ptr<LBInstruction> lb_instr = _lb_agent->getInstruction(_memory_manager);
    if (lb_instr != nullptr) {
        ManageOp op = lb_instr->op == SIGNAL_CACHE_IN? FETCH_MODEL: ERASE_MODEL;
        _queue_for_manage.push(shared_ptr<ManageInstruction>(new ManageInstruction(op, lb_instr->key, nullptr)));
    }
    return true;
}

void loopHandleQuery() {
    while (!_shutdown && onceHandleQuery())
        continue;
}

void loopManage() {
    while (!_shutdown && onceManage())
        continue;
}

void loopForStorage() {
    while (!_shutdown && onceForStorage())
        continue;
}

void loopForLB() {
    while (!_shutdown && onceForLB())
        continue;
}