#include <unistd.h>
#include <arpa/inet.h>

#include <string>
#include <iostream>

#include "lb_agent.h"

using namespace std;
using namespace balance::util;

LBAgent::LBAgent(string lb_addr, int lb_port, string addr_for_cli, int port_for_cli) {
    _client.reset(new TcpClient(lb_addr, lb_port));

    struct {
        int ip_int;
        int port;
    } location;
    inet_pton(AF_INET, addr_for_cli.c_str(), &(location.ip_int));
    location.port = port_for_cli;
    _client->tcpSend((char*)&location, sizeof(location));
}

LBAgent::~LBAgent() {

}

void LBAgent::completeTask(string model_name, char* data, size_t size) {
    _client->tcpSendWithLength(data, size);
}

string LBAgent::getTask() {
    string model_name;
    _client->tcpRecvString(model_name);
    return model_name;
}

pair<shared_ptr<char>, size_t> LBAgent::getData() {
    shared_ptr<char> data_ptr;
    size_t data_size;
    _client->tcpRecvWithLength(data_ptr, data_size);
    return make_pair<>(data_ptr, data_size);
}