#include <unistd.h>

#include <string>
#include <memory>
#include <iostream>
#include <unordered_map>

#include "utils/utils.h"

using namespace std;
using namespace balance::util;

struct Request {
    int op;
    string key;
    string value;
};

unordered_map<string, string> _storage;

Request getRequest(shared_ptr<TcpAgent> agent) {
    Request req;
    agent->tcpRecv((char*)&req.op, sizeof(req.op));
    agent->tcpRecvString(req.key);
    if (req.op == KVSTORAGE_OP_WRITE)
        agent->tcpRecvString(req.value);
    return req;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        cout << string("Argument Error") << endl;
        cout << string("program [ListenAddr] [ListenPort]") << endl;
        return 0;
    }
    
    string address(argv[1]);
    int port = stoi(argv[2]);

    TcpServer server(address, port);
    while (true) {
        shared_ptr<TcpAgent> agent = server.tcpAccept();
        cout << "Get connection" << endl;
        Request req = getRequest(agent);
        if (req.op == KVSTORAGE_OP_READ) {
            if (_storage.find(req.key) == _storage.end())
                agent->tcpSendString(string(""));
            else
                agent->tcpSendString(_storage[req.key]);
        }
        else {
            _storage[req.key] = req.value;
            agent->tcpSendString(string("ACK"));
        }
        cout << "Connection complete" << endl;
    }
    
    return 0;
}