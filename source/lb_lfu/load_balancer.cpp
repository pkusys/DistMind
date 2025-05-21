#include <unistd.h>
#include <string.h>

#include <string>
#include <memory>
#include <iostream>
#include <thread>
#include <queue>
#include <unordered_map>
#include <mutex>
#include <vector>

#include "utils/utils.h"

#include "controller_client.h"
#include "client_agent.h"
#include "server_agent.h"
#include "cache_agent.h"
#include "dispatcher.h"
#include "metadata_manager.h"

using namespace std;
using namespace balance::util;

struct AgentHandlerForClient: public AgentHandler {
    void run(shared_ptr<TcpAgent> agent);
};
struct AgentHandlerForServer: public AgentHandler {
    void run(shared_ptr<TcpAgent> agent);
};
struct AgentHandlerForCache: public AgentHandler {
    void run(shared_ptr<TcpAgent> agent);
};

shared_ptr<ControllerClient> _controller;
shared_ptr<MetadataManager> _metadata;
shared_ptr<Dispatcher<LBTask, LBTask> > _dispatcher;
vector<shared_ptr<thread> > _server_worker;
mutex lb_model_query_queue_lock;
queue<shared_ptr<LBTask> > lb_model_query_queue;

inline void enqueueQuery(shared_ptr<LBTask> t) {
    const lock_guard<mutex> lock(lb_model_query_queue_lock);
    lb_model_query_queue.push(t);
}

inline shared_ptr<LBTask> dequeueQuery() {
    const lock_guard<mutex> lock(lb_model_query_queue_lock);
    if (lb_model_query_queue.empty())
        return nullptr;
    else {
        shared_ptr<LBTask> t = lb_model_query_queue.front();
        lb_model_query_queue.pop();
        return t;
    }
}

void handleServerRegistration(shared_ptr<LBTask> t);
void handleCacheRegistration(shared_ptr<LBTask> t);
void handleQueryCompletion(shared_ptr<LBTask> t);
void handleRequest(shared_ptr<LBTask> t);
void handleRequestCompletion(shared_ptr<LBTask> t);

bool onceProcessTask();
void loopProcessTask();
bool onceQueryModel(string address, int port);
void loopQueryModel(string address, int port);

int main(int argc, char** argv) {
    if (argc < 9) {
        cout << string("Argument Error") << endl;
        cout << string("program [StrategyName] [AddrForClient] [PortForClient] [AddrForServer] [PortForServer] [CtrlAddr] [CtrlPort] [AddrForCache] [PortForCache] [MetadataStorageAddr] [MetadataStoragePort]") << endl;
        return 0;
    }
    
    string strategy_name(argv[1]);
    string address_for_client(argv[2]);
    int port_for_client = stoi(argv[3]);
    string address_for_cache(argv[4]);
    int port_for_cache = stoi(argv[5]);
    string address_for_server(argv[6]);
    int port_for_server = stoi(argv[7]);
    string controller_address(argv[8]);
    int controller_port = stoi(argv[9]);
    string metadata_storage_address(argv[10]);
    int metadata_storage_port = stoi(argv[11]);

    _controller.reset(new ControllerClient(controller_address, controller_port));
    cout << "Connect to the controller" << endl << endl;
    _metadata.reset(new MetadataManager());
    _dispatcher.reset(new Dispatcher<LBTask, LBTask>());
    TcpServerParallelWithAgentHandler s_client(address_for_client, port_for_client, shared_ptr<AgentHandler>(new AgentHandlerForClient()));
    TcpServerParallelWithAgentHandler s_server(address_for_server, port_for_server, shared_ptr<AgentHandler>(new AgentHandlerForServer()));
    TcpServerParallelWithAgentHandler s_cache(address_for_cache, port_for_cache, shared_ptr<AgentHandler>(new AgentHandlerForCache()));
    thread thd_query(loopQueryModel, metadata_storage_address, metadata_storage_port);
    thread thd_process(loopProcessTask);
    cout << "Threads started" << endl;

    while (true)
        sleep(1);
    
    return 0;
}

void AgentHandlerForClient::run(shared_ptr<TcpAgent> agent) {
    double start_time = time_now();
    shared_ptr<ClientAgent> client_agent(new ClientAgent(agent));
    _dispatcher->registerCustomer(client_agent->getId());

    shared_ptr<LBTask> request = client_agent->recvRequest();
    bool train = request->model_name.find("train") != string::npos;
    _dispatcher->push(request, train);
    shared_ptr<LBTask> response = _dispatcher->customerPopBlocked(client_agent->getId());
    client_agent->sendResponse(response);
    double end_time = time_now();

    _dispatcher->eraseCustomer(client_agent->getId());
    cout << "Request latency: " << fixed << (end_time - start_time) * 1000 << " ms" << endl << endl << endl;
}

void AgentHandlerForServer::run(shared_ptr<TcpAgent> agent) {
    shared_ptr<ServerAgent> server_agent = ServerAgent::createServerAgent(agent);
    _dispatcher->registerService(server_agent->getId());

    shared_ptr<LBTask> task(new LBTask());
    task->op = LB_REGISTER_SERVER;
    task->id = server_agent->getId();
    _dispatcher->push(task);
}

void AgentHandlerForCache::run(shared_ptr<TcpAgent> agent) {
    shared_ptr<CacheAgent> cache_agent = CacheAgent::createCacheAgent(agent);

    shared_ptr<LBTask> task(new LBTask());
    task->op = LB_REGISTER_CACHE;
    task->id = cache_agent->getId();
    task->data_size = cache_agent->getCapacity();
    _dispatcher->push(task);
}

void handleServerRegistration(shared_ptr<LBTask> task) {
    shared_ptr<ServerAgent> server_agent = ServerAgent::getServerAgent(task->id);
    _metadata->register_server(server_agent->getIp(), server_agent->getPort());

    _server_worker.push_back(shared_ptr<thread>(new thread(
        [](shared_ptr<ServerAgent> server_agent) -> void {
            while (true)
                server_agent->sendRequest(
                    _dispatcher->serviceFetchBlocked(
                        server_agent->getId()
                    )
                );
        },
        server_agent
    )));
    sleep(1);

    _server_worker.push_back(shared_ptr<thread>(new thread(
        [](shared_ptr<ServerAgent> server_agent) -> void {
            while (true) {
                auto response = server_agent->recvResponse();
                response->server_ip = server_agent->getIp();
                response->server_port = server_agent->getPort();
                _dispatcher->push(response);
                // TODO: Remove updating metadata in sub-threads
                // _metadata->decrease_workload(server_agent->getIp(), server_agent->getPort(), response->model_name);
            }
        },
        server_agent
    )));

    cout << "Handle server registration" << endl << endl << endl;
}

void handleCacheRegistration(shared_ptr<LBTask> task) {
    shared_ptr<CacheAgent> cache_agent = CacheAgent::getCacheAgent(task->id);
    _metadata->register_cache(cache_agent->getIp(), cache_agent->getCapacity());

    cout << "Handle cache registration" << endl << endl << endl;
}

void handleQueryCompletion(shared_ptr<LBTask> task) {
    _metadata->set_model_size(task->model_name, task->data_size);

    cout << "Handle query completion" << endl << endl << endl;
}

void handleRequest(shared_ptr<LBTask> request) {
    size_t model_size = _metadata->get_model_size(request->model_name);
    if (model_size == 0) {
        enqueueQuery(request);
        return;
    }
    size_t required_model_space = model_size * MEMORY_MANAGER_AMPLIFIER;
    // cout << "handleRequest: " << request->model_name << ", " << model_size << endl;
    cout << "Request model: " << request->model_name << endl;

    auto all_workload = _metadata->get_workload();
    // {
    //     const int SERVER_SATURATED_THRESHOLD = 4;
    //     int min_workload = SERVER_SATURATED_THRESHOLD;
    //     int ip_int = 0, port = 0;
    //     for (auto itr_ip = all_workload.begin(); itr_ip != all_workload.end(); ++itr_ip) {
    //         for (auto itr_port = itr_ip->second.begin(); itr_port != itr_ip->second.end(); ++itr_port) {
    //             if (itr_port->second < min_workload) {
    //                 ip_int = itr_ip->first;
    //                 port = itr_port->first;
    //                 min_workload = itr_port->second;
    //             }
    //         }
    //     }

    //     if (min_workload == SERVER_SATURATED_THRESHOLD) {
    //         bool train = request->model_name.find("train") != string::npos;
    //         _dispatcher->push(request, train);
    //         return;
    //     }
    // }

    // Select target Instance
    int selection_ip_int = 0;
    if ((selection_ip_int = _metadata->check_cache(request->model_name)) == 0) {
        cout << "Cache Miss" << endl;
        // Cache Out
        while ((selection_ip_int = _metadata->check_cache_space(required_model_space)) == 0) {
            auto cache_out = _metadata->cache_out_model();
            int ip_int = cache_out.first;
            string model_out = cache_out.second;
            if (model_out.length() == 0) {
                bool train = request->model_name.find("train") != string::npos;
                _dispatcher->push(request, train);
                return;
            }
            shared_ptr<CacheAgent> cache_agent_out = CacheAgent::getCacheAgent(((uint64_t)ip_int) << 32);
            cache_agent_out->cacheOut(model_out);
        }
        cout << "Cache out" << endl;
        // Cache In
        shared_ptr<CacheAgent> cache_agent_in = CacheAgent::getCacheAgent(((uint64_t)selection_ip_int) << 32);
        cache_agent_in->cacheIn(request->model_name);
        cout << "Cache In" << endl;
    }
    else {
        cout << "Cache Hit" << endl;
    }
    _metadata->cache_in_model(selection_ip_int, request->model_name);

    // Select target GPU
    auto workload = all_workload[selection_ip_int];
    int selection_port = workload.begin()->first;
    int min_load = workload.begin()->second;
    for (auto itr = workload.begin(); itr != workload.end(); ++itr) {
        if (itr->second < min_load)
            selection_port = itr->first;
    }
    uint64_t server_id = (((uint64_t)selection_ip_int) << 32) + ((uint64_t)selection_port);
    shared_ptr<ServerAgent> server_agent = ServerAgent::getServerAgent(server_id);
    _dispatcher->dispatcherPush(server_agent->getId(), request);
    _metadata->increase_workload(server_agent->getIp(), server_agent->getPort(), request->model_name);

    cout << "Handle request" << endl << endl << endl;
}

void handleRequestCompletion(shared_ptr<LBTask> response) {
    _metadata->decrease_workload(response->server_ip, response->server_port, response->model_name);
    _dispatcher->serviceComplete(response->id, response);

    cout << "Handle request completion" << endl << endl << endl;
}

bool onceProcessTask() {
    shared_ptr<LBTask> task = _dispatcher->dispatcherPop(_metadata->check_idle_server());
    if (task != nullptr) {
        // cout << "Get task: " << task->op << endl;
        switch (task->op) {
            case LB_REGISTER_SERVER:     handleServerRegistration(task); break;
            case LB_REGISTER_CACHE:      handleCacheRegistration(task);  break;
            case LB_QUERY_COMPLETE:      handleQueryCompletion(task);    break;
            case LB_TASK_REQUEST:        handleRequest(task);            break;
            case LB_TASK_RESPONSE:       handleRequestCompletion(task);  break;
        }
        // cout << "Switch complete" << endl;
    }
    else {
        usleep(1000);
    }
    return true;
}

void loopProcessTask() {
    while (true)
        onceProcessTask();
}

bool onceQueryModel(string address, int port) {
    shared_ptr<LBTask> request = dequeueQuery();
    
    if (request != nullptr) {
        shared_ptr<TcpClient> storage(new TcpClient(address, port));

        // TODO: Define metadata
        int op = KVSTORAGE_OP_READ;
        storage->tcpSend((char*)&op, sizeof(op));

        string metadata_key = request->model_name + string("-SIZE");
        storage->tcpSendString(metadata_key);

        string model_size_b;
        storage->tcpRecvString(model_size_b);
        size_t model_size = *(size_t*)model_size_b.c_str();
        
        shared_ptr<LBTask> task(new LBTask());
        task->op = LB_QUERY_COMPLETE;
        task->model_name = request->model_name;
        task->data_size = model_size;
        _dispatcher->push(task);
        _dispatcher->push(request);
    }
    else
        usleep(1000);
    
    return true;
}

void loopQueryModel(string address, int port) {
    while (true)
        onceQueryModel(address, port);
}