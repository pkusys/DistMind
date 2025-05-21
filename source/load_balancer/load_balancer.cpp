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
#include "cache_agent.h"
#include "server_agent.h"
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

void handleServerRegistration(shared_ptr<LBTask> t);
void handleCacheRegistration(shared_ptr<LBTask> t);
void handleRequest(shared_ptr<LBTask> t);
void handleRequestCompletion(shared_ptr<LBTask> t);

bool onceProcessTask();
void loopProcessTask();
bool onceUpdateModelLocation();
void loopUpdateModelLocation();

int main(int argc, char** argv) {
    if (argc < 9) {
        cout << string("Argument Error") << endl;
        cout << string("program [StrategyName] [AddrForClient] [PortForClient] [AddrForServer] [PortForServer] [CtrlAddr] [CtrlPort] [MetadataStorageAddr] [MetadataStoragePort]") << endl;
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
    _metadata.reset(new MetadataManager(metadata_storage_address, metadata_storage_port));
    _dispatcher.reset(new Dispatcher<LBTask, LBTask>());
    TcpServerParallelWithAgentHandler s_client(address_for_client, port_for_client, shared_ptr<AgentHandler>(new AgentHandlerForClient()));
    TcpServerParallelWithAgentHandler s_server(address_for_server, port_for_server, shared_ptr<AgentHandler>(new AgentHandlerForServer()));
    TcpServerParallelWithAgentHandler s_cache(address_for_cache, port_for_cache, shared_ptr<AgentHandler>(new AgentHandlerForCache()));
    thread thd_process(loopProcessTask);
    thread thd_update(loopUpdateModelLocation);
    cout << "Threads started" << endl;

    while (true)
        sleep(1);

    return 0;
}

void AgentHandlerForClient::run(shared_ptr<TcpAgent> agent) {
    // cout << "Get client connection" << endl;
    // double start_time = time_now();
    shared_ptr<ClientAgent> client_agent(new ClientAgent(agent));

    shared_ptr<LBTask> request = client_agent->recvRequest();
    if (request == nullptr) {
        cout << "Request error: " << client_agent->getId() << endl;
        return;
    }

    _dispatcher->registerCustomer(client_agent->getId());
    // cout << "Register client, " << client_agent->getId() << endl;

    request->time_receive = time_now();
    // cout << "Receive request, " << client_agent->getId() << endl;
    _dispatcher->push(request);
    // cout << "Enqueue request, " << client_agent->getId() << endl;
    shared_ptr<LBTask> response = _dispatcher->customerPopBlocked(client_agent->getId());
    response->time_dequeue = time_now();
    // cout << "Dequeue response, " << client_agent->getId() << endl;
    client_agent->sendResponse(response);
    response->time_reply = time_now();
    // cout << "Reply the client, " << client_agent->getId() << endl;
    // double end_time = time_now();

    _dispatcher->eraseCustomer(client_agent->getId());
    response->time_finalize = time_now();
    // cout << "Erase the client, " << client_agent->getId() << endl;
    // cout    << int((response->time_schedule - response->time_receive) * 1000) << "\t"
    //         << int((response->time_forward_0 - response->time_schedule) * 1000) << "\t"
    //         << int((response->time_forward - response->time_forward_0) * 1000) << "\t"
    //         << int((response->time_compute - response->time_forward) * 1000) << "\t"
    //         << int((response->time_dispatch - response->time_compute) * 1000) << "\t"
    //         << int((response->time_dequeue - response->time_dispatch) * 1000) << "\t"
    //         << int((response->time_finalize - response->time_dequeue) * 1000) << "\t"
    //         << int((response->time_reply - response->time_finalize) * 1000) << "\t"
    //         << endl;
}

void AgentHandlerForCache::run(shared_ptr<TcpAgent> agent) {
    shared_ptr<CacheAgent> cache_agent = CacheAgent::createCacheAgent(agent);

    shared_ptr<LBTask> task(new LBTask());
    task->op = LB_REGISTER_CACHE;
    task->id = cache_agent->getId();
    task->data_size = cache_agent->getCapacity();
    _dispatcher->push(task);
}

void AgentHandlerForServer::run(shared_ptr<TcpAgent> agent) {
    shared_ptr<ServerAgent> server_agent = ServerAgent::createServerAgent(agent);
    _dispatcher->registerService(server_agent->getId());

    shared_ptr<LBTask> task(new LBTask());
    task->op = LB_REGISTER_SERVER;
    task->id = server_agent->getId();
    _dispatcher->push(task);
}

void handleServerRegistration(shared_ptr<LBTask> task) {
    shared_ptr<ServerAgent> server_agent = ServerAgent::getServerAgent(task->id);
    _metadata->registerServer(server_agent->getIp(), server_agent->getPort());

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
            }
        },
        server_agent
    )));

    // cout << "Handle server registration" << endl << endl << endl;
}

void handleCacheRegistration(shared_ptr<LBTask> task) {
    shared_ptr<CacheAgent> cache_agent = CacheAgent::getCacheAgent(task->id);
    _metadata->registerCache(cache_agent->getIp(), cache_agent->getCapacity());

    for (auto model_info: _controller->getModels()) {
        if (_metadata->insertCache(cache_agent->getIp(), model_info.first))
            cache_agent->cacheIn(model_info.first);
        else
            break;
    }

    cout << "Handle cache registration" << endl << endl << endl;
}

void handleRequest(shared_ptr<LBTask> request) {
    // double start_time = time_now();
    int threshold = 8; // Only send reqeusts to GPUs whose waiting list is less than 2
    pair<int, int> selection = _metadata->getIdleServer(request->model_name, threshold);
    if (selection.first == 0 && selection.second == 0) {
        _dispatcher->push(request);
        cout << "Handle request: Miss" << ", " << request->id << endl;
        return;
    }

    uint64_t server_id = (((uint64_t)selection.first) << 32) + ((uint64_t)selection.second);
    shared_ptr<ServerAgent> server_agent = ServerAgent::getServerAgent(server_id);

    if (!_metadata->checkCache(server_agent->getIp(), request->model_name)) {
        uint64_t cache_id = (((uint64_t)server_agent->getIp()) << 32);
        shared_ptr<CacheAgent> cache_agent = CacheAgent::getCacheAgent(cache_id);
        while (! _metadata->insertCache(server_agent->getIp(), request->model_name)) {
            auto cache_set = _metadata->getCacheSet(server_agent->getIp());
            string model_out("");
            for (auto itr = cache_set.begin(); itr != cache_set.end(); ++itr) {
                if (_metadata->getActivity(server_agent->getIp(), *itr) == 0 && model_out.compare(*itr) < 0)
                    model_out = *itr;
            }
            if (model_out.compare("") == 0) {
                _dispatcher->push(request);
                cout << "Handle Request: Cache overflow" << endl;
                return;
            }
            _metadata->removeCache(server_agent->getIp(), model_out);
            cache_agent->cacheOut(model_out);
        }
        cache_agent->cacheIn(request->model_name);
    }

    _dispatcher->dispatcherPush(server_agent->getId(), request);
    _metadata->increaseWorkload(server_agent->getIp(), server_agent->getPort(), request->model_name);
    // cout << "Handle request" << endl;
    // double end_time = time_now();
    // cout << "\t\t\t\t" << fixed << (end_time - start_time) * 1000 << endl;
    request->time_schedule = time_now();
}

void handleRequestCompletion(shared_ptr<LBTask> response) {
    _metadata->decreaseWorkload(response->server_ip, response->server_port, response->model_name);
    _dispatcher->serviceComplete(response->id, response);
    response->time_dispatch = time_now();
    // cout << "Handle request completion" << endl;
}

bool onceProcessTask() {
    // Disable training currently
    // cout << 1 << endl;
    shared_ptr<LBTask> task = _dispatcher->dispatcherPop();
    // cout << 2 << endl;
    if (task != nullptr) {
        // cout << 3 << endl;
        // cout << "Get task: " << task->op << endl;
        switch (task->op) {
            case LB_REGISTER_SERVER:     handleServerRegistration(task); break;
            case LB_REGISTER_CACHE:      handleCacheRegistration(task);  break;
            case LB_TASK_REQUEST:        handleRequest(task);            break;
            case LB_TASK_RESPONSE:       handleRequestCompletion(task);  break;
        }
        // cout << "Switch complete" << endl;
    }
    else {
        usleep(10);
        // cout << "Sleep for 10us" << endl;
    }
    return true;
}

void loopProcessTask() {
    while (true)
        onceProcessTask();
}

bool onceUpdateModelLocation() {
    auto notification = _controller->getNotification();
    cout << "Get notification" << endl;
    _metadata->registerModel(notification.ip_int, notification.port, notification.model_name);
    cout << "Update server model" << ", " << (unsigned int)notification.ip_int << ", " << notification.port << ", " << notification.model_name << endl;
    return true;
}

void loopUpdateModelLocation() {
    while (true)
        onceUpdateModelLocation();
}