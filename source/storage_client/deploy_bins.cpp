#include "cli_common.hpp"
#include <unistd.h>
#include <fstream>
#include <unordered_map>

using namespace pipeps::store;

std::unordered_map<std::string, size_t> key_sizes;

// TODO
void init_store_clis(std::vector<StoreCli*>& clis,
                     std::vector<std::pair<std::string, std::string>>& ip_ports,
                     std::string shm_name,
                     size_t shm_size);

// TODO write data to loc, return model length
size_t load_full_model(std::string path, char* buf);

// TODO read batches into sptr, record each batch location in the vector
void load_batches(std::string folder,
                  std::string model_name,
                  char* buf_s,
                  size_t offset,
                  int num_batches,
                  std::vector<std::string>& keys,
                  std::vector<std::pair<size_t, size_t>>& data_loc);

// TODO save the entire model data into one storage server
void push_full_model(StoreCli* cli,
                     std::string model_name,
                     size_t offset,
                     size_t len);

// TODO push data batch stored in data, round robin way
void push_hori(std::vector<StoreCli*>& clis,
               std::vector<std::string>& keys,
               std::vector<std::pair<size_t, size_t>>& data);

// TODO push entire model to multiple storages evenly
// keys = [model-name-v1, model-name-v2 ...]
void push_verti(std::vector<StoreCli*>& clis,
                std::string model_name,
                size_t offset,
                size_t len);

// TODO push hybrid,
// using modified key: model-name-b0-v0 etc.
void push_hybrid(std::vector<StoreCli*>& clis,
                 std::vector<std::string>& keys,
                 std::vector<std::pair<size_t, size_t>>& data);

void write_out_key_size(std::string folder);

int main(int argc, char const* argv[]) {
  std::string temp_cache = "deploy_cache";
  size_t cache_size = 2 * 1024 * 1024 * 1024UL;

  std::vector<std::pair<std::string, std::string>> ip_ports;
  // 从settings/storage_list.txt中读取port为7778的IP地址
  load_storage_ips(ip_ports, "7778");
  
  // 如果没有找到任何服务器，使用默认配置
  if (ip_ports.empty()) {
    spdlog::warn("Using default storage servers configuration");
    ip_ports.push_back(std::make_pair("127.0.0.1", "7778"));
  }
  spdlog::info("Found {} storage servers", ip_ports.size());

  std::string model_folder("./build/test9");
  std::string model_name("resnet152");  //
  int num_batches = 8;                 // predefined
  std::string full_bin_path =
      model_folder + "/" + model_name + "-fullmodel.bin";
  spdlog::info("full model path {}", full_bin_path);

  void* buf_s = openSHM(temp_cache, cache_size);
  std::vector<StoreCli*> storeClients;
  init_store_clis(storeClients, ip_ports, temp_cache, cache_size);

  // load data
  size_t full_model_size = load_full_model(full_bin_path, (char*)buf_s);

  char* batch_buf = (char*)buf_s + full_model_size;

  std::vector<std::string> keys;
  std::vector<std::pair<size_t, size_t>> data_loc;
  load_batches(model_folder + "/layer_batches", model_name, (char*)buf_s,
               full_model_size, num_batches, keys, data_loc);

  // push model to first storage
  push_full_model(storeClients[0], model_name, 0, full_model_size);
  spdlog::info("Completed push full model");

  push_hori(storeClients, keys, data_loc);
  spdlog::info("Completed push horizontal batches");

  push_verti(storeClients, model_name, 0, full_model_size);
  spdlog::info("Completed vertical partition");

  push_hybrid(storeClients, keys, data_loc);
  spdlog::info("Completed hybrid");

  for (int i = 0; i < storeClients.size(); i++) {
    delete storeClients[i];
  }
  sleep(3);
  // at end
  shm_unlink(temp_cache.c_str());
  write_out_key_size(model_folder);
}

void init_store_clis(std::vector<StoreCli*>& clis,
                     std::vector<std::pair<std::string, std::string>>& ip_ports,
                     std::string shm_name,
                     size_t shm_size) {
  std::string cname = "deploy";
  int i = 0;
  for (auto p : ip_ports) {
    StoreCli* c = new StoreCli(cname + std::to_string(i), p.first, p.second,
                               shm_name, shm_size, 4);
    clis.push_back(c);
    i++;
  }
}

size_t load_full_model(std::string path, char* buf) {
  return _load_to(path, buf);
}

void load_batches(std::string folder,
                  std::string model_name,
                  char* buf_s,
                  size_t offset,
                  int num_batches,
                  std::vector<std::string>& keys,
                  std::vector<std::pair<size_t, size_t>>& data_loc) {
  for (int i = 0; i < num_batches; i++) {
    char* data_buf = buf_s + offset;
    std::string filename =
        folder + "/" + model_name + "-" + std::to_string(i) + ".bin";
    size_t param_size = _load_to(filename, data_buf);
    std::string k = model_name + "-h-" + std::to_string(i);
    keys.push_back(k);
    data_loc.push_back(std::make_pair(offset, param_size));
    offset += param_size;
  }
}

void push_full_model(StoreCli* cli,
                     std::string model_name,
                     size_t offset,
                     size_t len) {
  std::vector<std::pair<size_t, size_t>> sendDataLoc;
  std::string key = model_name + "-full-model";
  sendDataLoc.push_back(std::make_pair(offset, len));
  size_t curCntr = cli->getCommCntr();
  int target = curCntr + sendDataLoc.size();
  cli->push(key, sendDataLoc);

  while (cli->getCommCntr() != target) {
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  spdlog::info("Pushed entire model to first storage with key {}", key);
  key_sizes[key] = len;
}

void push_hori(std::vector<StoreCli*>& clis,
               std::vector<std::string>& keys,
               std::vector<std::pair<size_t, size_t>>& data) {
  int num_cli = clis.size();

  for (int i = 0; i < keys.size(); i++) {
    std::string key = keys[i];
    spdlog::info("Sending batch {} to storage {}", key, i % num_cli);
    StoreCli* c = clis[i % num_cli];
    std::vector<std::pair<size_t, size_t>> sendDataLoc;
    sendDataLoc.push_back(data[i]);
    size_t curCntr = c->getCommCntr();
    size_t target = curCntr + sendDataLoc.size();
    c->push(key, sendDataLoc);
    while (c->getCommCntr() != target) {
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }

    spdlog::info("Sent {} with size {} to {}", key, data[i].second,
                 i % num_cli);
    key_sizes[key] = data[i].second;
  }
}

void push_verti(std::vector<StoreCli*>& clis,
                std::string model_name,
                size_t offset,
                size_t len) {
  int num_cli = clis.size();
  size_t chunk_size = len / num_cli;
  for (int i = 0; i < num_cli - 1; i++) {
    std::string key = model_name + "-v-" + std::to_string(i);
    std::vector<std::pair<size_t, size_t>> sendDataLoc;
    size_t _off = offset + i * chunk_size;
    sendDataLoc.push_back(std::make_pair(_off, chunk_size));
    size_t curCntr = clis[i]->getCommCntr();
    size_t target = curCntr + sendDataLoc.size();
    clis[i]->push(key, sendDataLoc);
    while (clis[i]->getCommCntr() != target) {
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    spdlog::info("Sent {} to storage {}", key, i);
    key_sizes[key] = chunk_size;
  }

  size_t _off = offset + (num_cli - 1) * chunk_size;
  size_t remain = len - (num_cli - 1) * chunk_size;
  std::string key = model_name + "-v-" + std::to_string(num_cli - 1);
  std::vector<std::pair<size_t, size_t>> sendDataLoc;
  sendDataLoc.push_back(std::make_pair(_off, remain));
  size_t curCntr = clis[num_cli - 1]->getCommCntr();
  size_t target = curCntr + sendDataLoc.size();
  clis[num_cli - 1]->push(key, sendDataLoc);
  while (clis[num_cli - 1]->getCommCntr() != target) {
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  spdlog::info("Sent {} to storage {}", key, num_cli - 1);
  key_sizes[key] = remain;
}

void push_hybrid(std::vector<StoreCli*>& clis,
                 std::vector<std::string>& keys,
                 std::vector<std::pair<size_t, size_t>>& data) {
  int num_batches = keys.size();
  int num_clis = clis.size();
  for (int i = 0; i < num_batches; i++) {
    size_t param_size = data[i].second;
    size_t offset = data[i].first;

    size_t chunk_size = param_size / num_clis;

    for (int j = 0; j < num_clis; j++) {
      // split each batch onto multiple storages
      std::string _key = keys[i] + "-v-" + std::to_string(j);
      std::vector<std::pair<size_t, size_t>> sendDataLoc;
      size_t _s = chunk_size;
      if (j == num_clis - 1) {
        _s = param_size - (num_clis-1) * chunk_size;
      }
      size_t _off = offset+j*chunk_size;
      sendDataLoc.push_back(std::make_pair(_off, _s));

      size_t cntr = clis[j]->getCommCntr();
      size_t target = cntr + 1;
      clis[j]->push(_key, sendDataLoc);
      while (clis[j]->getCommCntr() != target) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
      }
      spdlog::info("Sent {} to storage {}", _key, j);
      key_sizes[_key] = _s;
    }
  }
}


void write_out_key_size(std::string folder){
  std::string filepath = folder + "/key_sizes.txt";
  std::fstream outfile;
  outfile.open(filepath, std::ios::out);
  if (outfile.is_open()) {
    for (auto p: key_sizes) {
      outfile << p.first << ", " << p.second << "\n";
    }
  }
  outfile.close();
}