#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <execinfo.h>
#include <signal.h>

#include <string>
#include <iostream>
#include <memory>
#include <unordered_map>

#include <pybind11/pybind11.h>

#include "utils/utils.h"

using namespace std;
namespace py = pybind11;
using namespace balance::util;
using namespace pipeps::store;

shared_ptr<SharedMemory> _shm;
unordered_map<string, shared_ptr<StoreCli> > _connections;

void pyInitialize(string shm_name, size_t shm_size);
void pyConnect(string cli_name, string addr, int port);
void pyPutKVBytes(string cli_name, string key, char* data, size_t size);
py::bytes pyGetKVBytes(string cli_name, string key, size_t size);
void pyFinalize();

void initialize(string shm_name, size_t shm_size);
void connect(string cli_name, string addr, int port);
string putKV(string cli_name, string key, char* data, size_t size);
string getKV(string cli_name, string key, char* data, size_t size);
void finalize();

PYBIND11_MODULE(deployment_c, m) {
    m.def("initialize", &pyInitialize, "pyInitialize");
    m.def("connect", &pyConnect, "pyConnect");
    m.def("put_kv_bytes", &pyPutKVBytes, "pyPutKVBytes");
    m.def("get_kv_bytes", &pyGetKVBytes, "pyGetKVBytes");
    m.def("finalize", &pyFinalize, "pyFinalize");
}

void pyInitialize(string shm_name, size_t shm_size) {
    initialize(shm_name, shm_size);
}
void pyConnect(string cli_name, string addr, int port) {
    connect(cli_name, addr, port);
}
void pyPutKVBytes(string cli_name, string key, char* data, size_t size) {
    putKV(cli_name, key, data, size);
}
py::bytes pyGetKVBytes(string cli_name, string key, size_t size) {
    char* data = (char*)malloc(size);
    getKV(cli_name, key, data, size);
    py::bytes data_b(data, size);
    free(data);
    return data_b;
}
void pyFinalize() {
    finalize();
}

void initialize(string shm_name, size_t shm_size) {
    _shm.reset(new SharedMemory(shm_name, shm_size, true));
}

void connect(string cli_name, string addr, int port) {
    shared_ptr<StoreCli> sCli(new StoreCli(cli_name, addr, to_string(port), _shm->getName(), _shm->getSize(), 4));
    _connections[cli_name] = sCli;
}

string putKV(string cli_name, string key, char* data, size_t size) {
  std::string _msg("putKV:: key:" + key + "\n");
  std::cout << _msg;
    if (_connections.find(cli_name) == _connections.end())
        return string("Not Connected");
    auto sCli = _connections[cli_name];
    
    vector<pair<size_t, size_t> > data_shm;
    data_shm.push_back(make_pair<>(0, size));
    memcpy(_shm->getPointer(), data, size);

    // Debug
    float *test = (float*)_shm->getPointer();
    float sum = 0.0;
    for (size_t i = 0;i < size / sizeof(float); ++i)
        sum += test[i];
    cout << "Data before put in C++: " << sum << endl;
    // Debug End

    int target = sCli->getCommCntr() + 1;
    sCli->push(key, data_shm);
    while (sCli->getCommCntr() != target)
        usleep(1000);

    return string("OK");
}

string getKV(string cli_name, string key, char* data, size_t size) {
    if (_connections.find(cli_name) == _connections.end())
        return string("Not Connected");
    auto sCli = _connections[cli_name];

    vector<pair<size_t, size_t> > data_shm;
    data_shm.push_back(make_pair<>(0, size));

    int target = sCli->getCommCntr() + 1;
    sCli->pull(key, data_shm);
    while (sCli->getCommCntr() != target)
        usleep(1000);

    // Debug
    float *test = (float*)_shm->getPointer();
    float sum = 0.0;
    for (size_t i = 0; i < size / sizeof(float); ++i)
        sum += test[i];
    cout << "Data after get in C++: " << sum << endl;
    // Debug End

    memcpy(data, _shm->getPointer(), size);

    return string("OK");
}

void finalize() {
    
}