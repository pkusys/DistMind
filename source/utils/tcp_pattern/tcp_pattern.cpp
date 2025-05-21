#include <unistd.h>

#include <string>
#include <thread>

#include "tcp_pattern.h"

using namespace std;

namespace balance {
namespace util {

TcpServerParallelWithAgentHandler::TcpServerParallelWithAgentHandler(string address, int port, std::shared_ptr<AgentHandler> handler) {
    _server.reset(new TcpServer(address, port, 64));
    _handler = handler;
    _listener.reset(new thread(
        [](shared_ptr<TcpServer> s, shared_ptr<AgentHandler> h) -> void {
            while (true) {
                shared_ptr<TcpAgent> agent = s->tcpAccept();
                thread t(
                    [](shared_ptr<AgentHandler> h, shared_ptr<TcpAgent> agent) -> void {h->run(agent);},
                    h, agent
                );
                t.detach();
            }
        }, 
        _server, _handler
    ));
}

TcpServerParallelWithAgentHandler::~TcpServerParallelWithAgentHandler() {
    _listener->join();
}

} //namespace util
} //namespace balance