#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <execinfo.h>
#include <signal.h>

#include <string>
#include <iostream>
#include <fstream>

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "utils/utils.h"

using namespace std;
namespace py = pybind11;
using namespace torch;
using namespace balance::util;

int _model_counter;
ofstream _kv_file;

void pyInitialize(string filename);
void pyPutModelProfile(string model_name, size_t model_size, int num_batches);
void pyPutKVTensor(string key, Tensor t);
void pyPutKVBytes(string key, char* data, size_t size);
void pyFinalize();

void initialize(string filename);
void putModelProfile(string model_name, size_t model_size, int num_batches);
void putKV(string key, char* data, size_t size);
void getKV(string key, char* data, size_t size);
void finalize();

PYBIND11_MODULE(deploy_generate_c, m) {
    m.def("initialize", &pyInitialize, "pyInitialize");
    m.def("put_model_profile", &pyPutModelProfile, "pyPutModelProfile");
    m.def("put_kv_tensor", &pyPutKVTensor, "pyPutKVTensor");
    m.def("put_kv_bytes", &pyPutKVBytes, "pyPutKVBytes");
    m.def("finalize", &pyFinalize, "pyFinalize");
}

void pyInitialize(string filename) {
    initialize(filename);
}
void pyPutModelProfile(string model_name, size_t model_size, int num_batches) {
    putModelProfile(model_name, model_size, num_batches);
}
void pyPutKVTensor(string key, Tensor t) {
    char* data = (char*)t.data_ptr();
    size_t size = t.numel() * t.element_size();
    return putKV(key, data, size);
}
void pyPutKVBytes(string key, char* data, size_t size) {
    putKV(key, data, size);
}
void pyFinalize() {
    finalize();
}

void initialize(string filename) {
    _model_counter = 0;
    _kv_file.open(filename, ios::out | ios::binary);
    _kv_file.write((char*)&_model_counter, sizeof(_model_counter));
}

void putModelProfile(string model_name, size_t model_size, int num_batches) {
    ++_model_counter;
    
    size_t model_name_size = model_name.size();

    _kv_file.write((char*)&model_name_size, sizeof(model_name_size));
    _kv_file.write(model_name.c_str(), model_name_size);
    _kv_file.write((char*)&model_size, sizeof(model_size));
    _kv_file.write((char*)&num_batches, sizeof(num_batches));
}

void putKV(string key, char* data, size_t size) {
    size_t kv_size = 0;
    size_t key_size = key.size();
    kv_size += (sizeof(key_size) + key_size);
    kv_size += (sizeof(size) + size);

    _kv_file.write((char*)&kv_size, sizeof(kv_size));
    _kv_file.write((char*)&key_size, sizeof(key_size));
    _kv_file.write(key.c_str(), key_size);
    _kv_file.write((char*)&size, sizeof(size));
    _kv_file.write(data, size);
}

void finalize() {
    _kv_file.seekp(0, ios::beg);
    _kv_file.write((char*)&_model_counter, sizeof(_model_counter));
    _kv_file.close();
}