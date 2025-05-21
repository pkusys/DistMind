#include <stdlib.h>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include "cli_common.hpp"
#include "spdlog/spdlog.h"
#include "utils/tcp/tcp.h"

using namespace balance::util;
std::string model_name = "resnet152";
std::string log_dir = "./tmp/test9/";
int num_batches = 8;

void load_sizes(std::string filename,
                std::unordered_map<std::string, size_t>& key_sizes);

// TODO
void per_app_exp(std::vector<std::shared_ptr<TcpAgent>>& pull_clis,
                 std::vector<std::string>& storeIPs,
                 std::unordered_map<std::string, size_t>& key_sizes,
                 int n_repeat);

// TODO
void hori_exp(std::vector<std::shared_ptr<TcpAgent>>& pull_clis,
              std::vector<std::string>& storeIPs,
              std::unordered_map<std::string, size_t>& key_sizes,
              int n_repeat);

void verti_exp(std::vector<std::shared_ptr<TcpAgent>>& pull_clis,
               std::vector<std::string>& storeIPs,
               std::unordered_map<std::string, size_t>& key_sizes,
               int n_repeat);

void hybrid_exp(std::vector<std::shared_ptr<TcpAgent>>& pull_clis,
                std::vector<std::string>& storeIPs,
                std::unordered_map<std::string, size_t>& key_sizes,
                int n_repeat);

std::tuple<double, double, double> avg_min_max(std::vector<double>& values);

void write_out_vals(std::string filename, std::vector<double>& values) {
  std::fstream outfile;
  outfile.open(filename, std::ios::out);
  if (outfile.is_open()) {
    for (auto v : values) {
      outfile << v << "\n";
    }
    outfile.close();
  }
}

int main(int argc, char const* argv[]) {
  spdlog::set_level(spdlog::level::debug);
  if (argc < 3) {
    spdlog::error("usage: ./shard_ctl <num-clis> <cntr_port>");
  }
  int num_clis = atoi(argv[1]);
  std::string sizes_file =
      "./build/test9/key_sizes.txt";
  std::unordered_map<std::string, size_t> key_sizes;
  load_sizes(sizes_file, key_sizes);

  int server_port = atoi(argv[2]);
  std::vector<std::string> storeIPs;
  std::vector<std::pair<std::string, std::string>> ip_ports;
  // 从settings/storage_list.txt中读取port为7778的IP地址
  load_storage_ips(ip_ports, "7778");
  
  // 如果没有找到任何服务器，使用默认配置
  if (ip_ports.empty()) {
    spdlog::warn("Using default storage servers configuration");
    ip_ports.push_back(std::make_pair("127.0.0.1", "7778"));
  }
  for (auto p : ip_ports) {
    storeIPs.push_back(p.first);
  }

  TcpServer ctl_server("0.0.0.0", server_port);
  std::vector<std::shared_ptr<TcpAgent>> pull_clis;

  for (int i = 0; i < num_clis; i++) {
    std::shared_ptr<TcpAgent> c = ctl_server.tcpAccept();
    pull_clis.push_back(c);
    spdlog::debug("accepted one test client");
  }

  sleep(1);

  spdlog::info("per_app_exp ======================");
  per_app_exp(pull_clis, storeIPs, key_sizes, 10);
  per_app_exp(pull_clis, storeIPs, key_sizes, 10);
  spdlog::info("hori_exp ======================");
  hori_exp(pull_clis, storeIPs, key_sizes, 10);
  hori_exp(pull_clis, storeIPs, key_sizes, 10);
  spdlog::info("verti_exp ======================");
  verti_exp(pull_clis, storeIPs, key_sizes, 10);
  verti_exp(pull_clis, storeIPs, key_sizes, 10);
  spdlog::info("hybrid_exp ======================");
  hybrid_exp(pull_clis, storeIPs, key_sizes, 10);
  hybrid_exp(pull_clis, storeIPs, key_sizes, 10);

  for (auto c : pull_clis) {
    std::string exit("exit");
    c->tcpSendWithLength(exit.c_str(), exit.size());
  }
  sleep(1);
}

void load_sizes(std::string filename,
                std::unordered_map<std::string, size_t>& key_sizes) {
  std::fstream infile;
  infile.open(filename, std::ios::in);
  if (infile.is_open()) {
    std::string line;
    while (getline(infile, line)) {
      size_t pos = line.find(",");
      std::string key = line.substr(0, pos);
      std::string size_str = line.substr(pos + 1, line.length());
      size_t size = std::stoull(size_str);
      key_sizes[key] = size;
    }
    infile.close();
  } else {
    spdlog::debug("error while opening key_sizes.txt");
  }
}

void per_app_exp(std::vector<std::shared_ptr<TcpAgent>>& pull_clis,
                 std::vector<std::string>& storeIPs,
                 std::unordered_map<std::string, size_t>& key_sizes,
                 int n_repeat) {
  // select pull model-name-full-model from first IP
  std::string key = model_name + "-full-model";
  auto got = key_sizes.find(key);
  if (got == key_sizes.end()) {
    spdlog::error("key {} not found in key_sizes", key);
    return;
  }
  std::vector<double> latencies;

  for (int i = 0; i < n_repeat; i++) {
    // send instr
    for (auto c : pull_clis) {
      std::string instr = storeIPs[0] + delimiter + key + delimiter +
                          std::to_string(key_sizes[key]);
      c->tcpSendWithLength(instr.c_str(), instr.length());
    }
    // receive timing
    for (auto c : pull_clis) {
      std::shared_ptr<char> data_ptr;
      size_t data_size;
      c->tcpRecvWithLength(data_ptr, data_size);
      double time = atof(data_ptr.get());
      latencies.push_back(time);
      spdlog::debug("takes {} ms to finish", time);
    }
  }
  std::tuple<double, double, double> res = avg_min_max(latencies);
  spdlog::info("per_app_exp:: average (ms)[avg, min, max]: {}, {}, {}",
               std::get<0>(res), std::get<1>(res), std::get<2>(res));
  write_out_vals(log_dir + "per_app.txt", latencies);
}

void hori_exp(std::vector<std::shared_ptr<TcpAgent>>& pull_clis,
              std::vector<std::string>& storeIPs,
              std::unordered_map<std::string, size_t>& key_sizes,
              int n_repeat) {
  int num_stores = storeIPs.size();
  // double each_batch[num_batches] = {0};
  std::vector<std::vector<double>> each_batch;
  for (int i = 0; i < num_batches; i++) {
    std::vector<double> latency_con;
    each_batch.push_back(latency_con);
  }

  for (int i = 0; i < n_repeat; i++) {
    for (int j = 0; j < num_batches; j++) {
      int sidx = j % num_stores;
      std::string key = model_name + "-h-" + std::to_string(j);
      auto got = key_sizes.find(key);
      if (got == key_sizes.end()) {
        spdlog::error("key {} not found in key_sizes", key);
        return;
      }
      std::string instr = storeIPs[sidx] + delimiter + key + delimiter +
                          std::to_string(key_sizes[key]);
      // send
      for (auto c : pull_clis) {
        c->tcpSendWithLength(instr.c_str(), instr.size());
      }
      // recv time
      for (auto c : pull_clis) {
        std::shared_ptr<char> data_ptr;
        size_t data_size;
        c->tcpRecvWithLength(data_ptr, data_size);
        double time = atof(data_ptr.get());
        // each_batch[j] += time;
        each_batch[j].push_back(time);
        spdlog::debug("takes {} ms to finish", time);
      }
    }
  }
  for (int i = 0; i < num_batches; i++) {
    std::tuple<double, double, double> res = avg_min_max(each_batch[i]);
    spdlog::info("hori_exp::batch {} cost (ms)[avg, min, max]: {}, {}, {}", i,
                 std::get<0>(res), std::get<1>(res), std::get<2>(res));
    write_out_vals(log_dir + "hori-h-" + std::to_string(i) + ".txt",
                   each_batch[i]);
  }
}

void verti_exp(std::vector<std::shared_ptr<TcpAgent>>& pull_clis,
               std::vector<std::string>& storeIPs,
               std::unordered_map<std::string, size_t>& key_sizes,
               int n_repeat) {
  int num_store = storeIPs.size();
  // double total_time = 0;
  std::vector<double> latencies;
  for (int i = 0; i < n_repeat; i++) {
    //
    std::vector<std::string> inst_vec;
    for (int j = 0; j < num_store; j++) {
      inst_vec.push_back(storeIPs[j]);
      std::string key = model_name + "-v-" + std::to_string(j);
      auto got = key_sizes.find(key);
      if (got == key_sizes.end()) {
        spdlog::error("key {} not found in key_sizes", key);
        return;
      }
      inst_vec.push_back(key);
      inst_vec.push_back(std::to_string(key_sizes[key]));
    }
    std::stringstream ss;
    for (int x = 0; x < inst_vec.size() - 1; x++) {
      ss << inst_vec[x] << delimiter;
    }
    ss << inst_vec[inst_vec.size() - 1];

    std::string compound_instr = ss.str();
    spdlog::debug("verti_exp:: compound instr {}", compound_instr);
    // start pull
    for (auto c : pull_clis) {
      c->tcpSendWithLength(compound_instr.c_str(), compound_instr.size());
    }
    // wait for
    for (auto c : pull_clis) {
      std::shared_ptr<char> data_ptr;
      size_t data_size;
      c->tcpRecvWithLength(data_ptr, data_size);
      double time = atof(data_ptr.get());
      // total_time += time;
      latencies.push_back(time);
      spdlog::debug("verti_exp::Takes {} ms to finish", time);
    }
  }
  std::tuple<double, double, double> res = avg_min_max(latencies);
  spdlog::info("verti_exp:: takes (ms)[avg, min, max]: {}, {}, {}",
               std::get<0>(res), std::get<1>(res), std::get<2>(res));
  write_out_vals(log_dir + "verti_exp.txt", latencies);
}

void hybrid_exp(std::vector<std::shared_ptr<TcpAgent>>& pull_clis,
                std::vector<std::string>& storeIPs,
                std::unordered_map<std::string, size_t>& key_sizes,
                int n_repeat) {
  std::vector<std::vector<double>> each_batch;
  for (int i = 0; i < num_batches; i++) {
    std::vector<double> latency_con;
    each_batch.push_back(latency_con);
  }

  for (int i = 0; i < n_repeat; i++) {
    for (int j = 0; j < num_batches; j++) {
      std::vector<std::string> inst_vec;
      // for each batch create compound instruction
      for (int k = 0; k < storeIPs.size(); k++) {
        inst_vec.push_back(storeIPs[k]);
        std::string key =
            model_name + "-h-" + std::to_string(j) + "-v-" + std::to_string(k);
        auto got = key_sizes.find(key);
        if (got == key_sizes.end()) {
          spdlog::error("key {} not found in key_sizes", key);
          return;
        }
        inst_vec.push_back(key);
        inst_vec.push_back(std::to_string(key_sizes[key]));
      }
      std::stringstream ss;
      for (int x = 0; x < inst_vec.size() - 1; x++) {
        ss << inst_vec[x] << delimiter;
      }
      ss << inst_vec[inst_vec.size() - 1];

      std::string compound_instr = ss.str();

      // send
      for (auto c : pull_clis) {
        c->tcpSendWithLength(compound_instr.c_str(), compound_instr.size());
      }
      // wait for time of each batch
      for (auto c : pull_clis) {
        std::shared_ptr<char> data_ptr;
        size_t data_size;
        c->tcpRecvWithLength(data_ptr, data_size);
        double time = atof(data_ptr.get());
        // each_batch[j] += time;
        each_batch[j].push_back(time);
        spdlog::debug("hybrid_exp::batch {} Takes {} ms to finish", j, time);
      }
    }
  }

  for (int i = 0; i < num_batches; i++) {
    std::tuple<double, double, double> res = avg_min_max(each_batch[i]);
    spdlog::info("hybrid_exp::batch {} takes (ms)[avg, min, max]: {}, {}, {}",
                 i, std::get<0>(res), std::get<1>(res), std::get<2>(res));
    write_out_vals(log_dir + "hybrid-h-" + std::to_string(i) + ".txt", each_batch[i]);
  }
}

std::tuple<double, double, double> avg_min_max(std::vector<double>& values) {
  double total = 0;
  double min = values[0];
  double max = values[0];
  for (auto v : values) {
    if (v < min) {
      min = v;
    }
    if (v > max) {
      max = v;
    }
    total += v;
  }
  double avg = total / values.size();
  return std::make_tuple(avg, min, max);
}