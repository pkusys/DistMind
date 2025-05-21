#include <unistd.h>
#include <memory>
#include <unordered_map>
#include "cli_common.hpp"
#include "utils/tcp/tcp.h"

using namespace pipeps::store;
using namespace balance::util;
std::string base_name = "shard_test_cli";

// TODO
void init_store_clis(
    std::vector<std::pair<std::string, std::string>>& store_ip_ports,
    std::unordered_map<std::string, StoreCli*>& clis,
    std::unordered_map<std::string, size_t>& offsets,
    size_t shm_size
  );

// TODO
void parse_task(
    std::string task_str,
    std::vector<std::tuple<std::string, std::string, size_t>>& tasks);

// TODO return sum of each store clients
size_t get_total_cntr(std::unordered_map<std::string, StoreCli*>& storeClients);

// TODO launch
void launch_one_task(std::unordered_map<std::string, StoreCli*>& storeClients,
                     std::unordered_map<std::string, size_t>& offsets,
                     std::tuple<std::string, std::string, size_t>& task);

void clean_res(std::unordered_map<std::string, StoreCli*>& storeClients);

int main(int argc, char const* argv[]) {
  if (argc < 4) {
    spdlog::error("require client name prefix, ./shard_cli <name> <cntr_ip> <cntr_port>");
    return -1;
  }
  base_name = std::string(argv[1]) + "-" + base_name;
  spdlog::set_level(spdlog::level::debug);
  std::string ctl_ip = std::string(argv[2]);
  int ctl_port = std::stoi(std::string(argv[3]));
  unsigned int size = 20;
  if(argc > 4) {
    size = std::stoi(std::string(argv[4]));
  }
  size_t shm_size = 1024 * 1024 * 1024UL * size;

  std::vector<std::pair<std::string, std::string>> store_ip_ports;
  // 从settings/storage_list.txt中读取port为7778的IP地址
  load_storage_ips(store_ip_ports, "7778");
  
  // 如果没有找到任何服务器，使用默认配置
  if (store_ip_ports.empty()) {
    spdlog::warn("Using default storage servers configuration");
    store_ip_ports.push_back(std::make_pair("127.0.0.1", "7778"));
  }

  std::unordered_map<std::string, StoreCli*> storeClients;
  std::unordered_map<std::string, size_t> cliOffsets;
  init_store_clis(store_ip_ports, storeClients, cliOffsets, shm_size);

  // wait for ctl start
  sleep(30);
  // TODO
  TcpClient fromCtl(ctl_ip, ctl_port);
  while (true) {
    std::shared_ptr<char> data_ptr;
    size_t data_size;
    fromCtl.tcpRecvWithLength(data_ptr, data_size);
    std::string tasks_str(data_ptr.get(), data_size);
    if (tasks_str == "exit") {
      break;
    }
    std::vector<std::tuple<std::string, std::string, size_t>> tasks;
    parse_task(tasks_str, tasks);

    double s = trans::time_now();
    // TODO each task wait for 1
    size_t cur_cntr = get_total_cntr(storeClients);
    size_t target = cur_cntr + tasks.size();
    for (auto t : tasks) {
      launch_one_task(storeClients, cliOffsets, t);
    }

    while (get_total_cntr(storeClients) != target) {
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    double e = trans::time_now();
    spdlog::debug("tasks {:s} costs: {:f}", tasks_str, e - s);
    // report time cost
    std::string ret_msg = std::to_string((e - s) * 1000);
    fromCtl.tcpSendWithLength(ret_msg.c_str(), ret_msg.size());
  }
  clean_res(storeClients);
}

void init_store_clis(
    std::vector<std::pair<std::string, std::string>>& store_ip_ports,
    std::unordered_map<std::string, StoreCli*>& clis,
    std::unordered_map<std::string, size_t>& offsets,
    size_t shm_size
  ) {
  int i = 0;
  for (auto p : store_ip_ports) {
    std::string cli_name = base_name + "-" + std::to_string(i);
    std::string shm_name = cli_name + "-shm";
    void* dataPtr = openSHM(shm_name, shm_size);

    StoreCli* c =
        new StoreCli(cli_name, p.first, p.second, shm_name, shm_size, 4);
    clis[p.first] = c;
    offsets[cli_name] = 0;
    i++;
  }
}

void parse_task(
    std::string task_str,
    std::vector<std::tuple<std::string, std::string, size_t>>& tasks) {
  std::vector<std::string> tokens;
  size_t pos = 0;
  while ((pos = task_str.find(delimiter)) != std::string::npos) {
    std::string t = task_str.substr(0, pos);
    tokens.push_back(t);
    task_str.erase(0, pos + delimiter.length());
    spdlog::debug("get token {:s}", t);
  }
  tokens.push_back(task_str);
  spdlog::debug("last token {:s}", task_str);

  for (int i = 0; i < tokens.size() / 3; i++) {
    auto t = std::make_tuple(tokens[i * 3], tokens[i * 3 + 1],
                             std::stoull(tokens[i * 3 + 2]));
    tasks.push_back(t);
  }
}

size_t get_total_cntr(
    std::unordered_map<std::string, StoreCli*>& storeClients) {
  size_t ret = 0;
  for (auto it : storeClients) {
    ret += it.second->getCommCntr();
  }
  return ret;
}

void launch_one_task(std::unordered_map<std::string, StoreCli*>& storeClients,
                     std::unordered_map<std::string, size_t>& offsets,
                     std::tuple<std::string, std::string, size_t>& task) {
  StoreCli* c = storeClients[std::get<0>(task)];
  std::string key = std::get<1>(task);
  size_t data_size = std::get<2>(task);
  size_t shm_offset = offsets[c->name];

  std::vector<std::pair<size_t, size_t>> recvLoc;
  recvLoc.push_back(std::make_pair(shm_offset, data_size));
  offsets[c->name] += data_size;
  c->pull(key, recvLoc);
}

void clean_res(std::unordered_map<std::string, StoreCli*>& storeClients) {
  for (auto p : storeClients) {
    delete p.second;
  }
  sleep(1);
  for (int i = 0; i < storeClients.size(); i++) {
    std::string cli_name = base_name + "-" + std::to_string(i);
    std::string shm_name = cli_name + "-shm";
    shm_unlink(shm_name.c_str());
  }
}