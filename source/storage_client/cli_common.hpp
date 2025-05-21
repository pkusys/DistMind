#ifndef CLI_COMM_H
#define CLI_COMM_H
#include <fstream>
#include "spdlog/spdlog.h"
#include "utils/utils.h"
#include <sstream>

std::string delimiter = ";;";

void* openSHM(std::string& shmName, size_t& shmSize) {
  int data_fd = shm_open(shmName.c_str(), O_CREAT | O_RDWR, 0666);
  pipeps::store::check_err((ftruncate(data_fd, shmSize) < 0),
                           "ftruncate instr_fd err\n");
  void* data_buf_ptr =
      mmap(0, shmSize, PROT_READ | PROT_WRITE, MAP_SHARED, data_fd, 0);
  return data_buf_ptr;
};

void reportBandwidth(size_t& len, double& s, double& e) {
  double dur = e - s;
  double bw = ((len * 8) / dur) / 1e9;
  spdlog::info("bw: {:f} Gbps; dur: {:f}", bw, dur);
}

void gen_random(char* s, const int len) {
  static const char alphanum[] =
      "0123456789"
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      "abcdefghijklmnopqrstuvwxyz";

  for (int i = 0; i < len; ++i) {
    s[i] = alphanum[rand() % (sizeof(alphanum) - 1)];
  }

  s[len] = 0;
}

size_t _load_to(std::string& filename, char* data_buf) {
  std::ifstream is(filename, std::ifstream::binary);
  if (is) {
    is.seekg(0, is.end);
    size_t length = is.tellg();
    is.seekg(0, is.beg);

    std::cout << "Read " << filename << "\n";

    is.read(data_buf, length);
    if (is)
      std::cout << "all characters read successfully.\n";
    else
      std::cout << "error: only " << is.gcount() << " could be read";
    is.close();

    return length;
  } else {
    return -1;
  }
};

// Load storage server IP addresses from settings/storage_list.txt
void load_storage_ips(std::vector<std::pair<std::string, std::string>>& ip_ports, 
                      const std::string& port) {
  std::ifstream file("settings/storage_list.txt");
  if (!file.is_open()) {
    spdlog::error("Failed to open settings/storage_list.txt");
    return;
  }
  
  // Skip header line
  std::string line;
  std::getline(file, line);
  
  // Read each line
  while (std::getline(file, line)) {
    std::istringstream iss(line);
    std::string ip, p;
    spdlog::info("Reading line: {}", line);
    if (std::getline(iss, ip, ',') && std::getline(iss, p, ',')) {
      // Trim whitespace
      ip.erase(0, ip.find_first_not_of(" \t"));
      ip.erase(ip.find_last_not_of(" \t") + 1);
      p.erase(0, p.find_first_not_of(" \t"));
      p.erase(p.find_last_not_of(" \t") + 1);
      
      spdlog::info("Parsed IP: {}, Port: {}", ip, p);
      // Add only if port matches
      if (p == port) {
        ip_ports.push_back(std::make_pair(ip, p));
        spdlog::info("Added storage server: {}:{}", ip, p);
      }
    }
  }
  
  if (ip_ports.empty()) {
    spdlog::warn("No storage servers found with port {}", port);
  }
}

#endif