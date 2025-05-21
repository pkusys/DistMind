#include <unordered_map>
#include "cli_common.hpp"

using namespace pipeps::store;

size_t batchSize = 20 * 1024 * 1024;
size_t recvSize = 30 * 1024 * 1024;

// generate random data on shm
// return a random key length 10 char
std::string randData(size_t offset, size_t len);

void pushTest(StoreCli& cli,
              size_t& offset,
              size_t& cacheSize,
              std::unordered_map<std::string, size_t>& dataRecords);

void pullTest(void* dataPtr,
              StoreCli& cli,
              size_t& offset,
              size_t& cacheSize,
              std::unordered_map<std::string, size_t>& dataRecords,
              bool verification);

void dataVeri(void* mem, size_t offset1, size_t offset2, size_t len);

int main(int argc, char const* argv[]) {
  spdlog::info("storage stress test!");
  if (argc < 5) {
    spdlog::error("Usage: ./storage_stress_test <testName> <dstIp> <dstPort> <cacheSize>");
  }
  std::string name(argv[1]);
  std::string ip(argv[2]);
  std::string port(argv[3]);
  size_t cacheSize = std::stoull(argv[4]);  // this total size of shm
  size_t offset = 0;
  std::string cname(name + "-cli");

  std::string cacheName(name + "-cache");
  void* dataPtr = openSHM(cacheName, cacheSize);

  StoreCli cli(cname, ip, port, cacheName, cacheSize, 4);
  std::unordered_map<std::string, size_t> dataRecords;

  pushTest(cli, offset, cacheSize, dataRecords);
  spdlog::info("num of records: {}", dataRecords.size());
  int _pause;
  std::cin >> _pause; 
  pullTest(dataPtr, cli, offset, cacheSize, dataRecords, false);
  return 0;
}

void pushTest(StoreCli& cli,
              size_t& offset,
              size_t& cacheSize,
              std::unordered_map<std::string, size_t>& dataRecords) {
  size_t nBatch = (cacheSize - recvSize) / batchSize;
  spdlog::info("cacheSize: {}; recvSize {}; nBatch {};", cacheSize, recvSize, nBatch);
  int _pause;
  std::cin >> _pause;
  // push test
  for (size_t i = 0; i < nBatch; i++) {
    // construct push data
    std::string key = randData(offset, batchSize);
    std::vector<std::pair<size_t, size_t>> sendDataLoc;
    double s = trans::time_now();
    size_t curCntr = cli.getCommCntr();
    sendDataLoc.push_back(
        std::make_pair(offset, batchSize));  // push as single batch data
    cli.push(key, sendDataLoc);
    spdlog::debug("pushed key {:s}, with size {:d}", key, batchSize);
    while (cli.getCommCntr() != curCntr + 1) {
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    double e = trans::time_now();
    reportBandwidth(batchSize, s, e);
    dataRecords[key] = offset;
    offset += batchSize;
  }
}

void pullTest(void* dataPtr,
              StoreCli& cli,
              size_t& offset,
              size_t& cacheSize,
              std::unordered_map<std::string, size_t>& dataRecords,
              bool verification) {
  size_t pageSize = 4 * 1024 * 1024;
  std::vector<std::pair<size_t, size_t>> recvDataLoc;
  size_t n = batchSize / pageSize;
  size_t accSize = 0;
  for (int i = 0; i < n; i++) {
    size_t _off = offset + i * pageSize;
    recvDataLoc.push_back(std::make_pair(_off, pageSize));
    accSize += pageSize;
  }
  if (accSize < batchSize) {
    size_t _off = offset + accSize;
    recvDataLoc.push_back(std::make_pair(_off, batchSize - accSize));
  }

  for (auto item : dataRecords) {
    double s = trans::time_now();
    size_t curCntr = cli.getCommCntr();
    size_t targetCntr = curCntr + recvDataLoc.size();
    std::string key(item.first);
    cli.pull(key, recvDataLoc);
    while (cli.getCommCntr() != targetCntr) {
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    double e = trans::time_now();
    reportBandwidth(batchSize, s, e);
    if (verification) {
      dataVeri(dataPtr, item.second, offset, batchSize);
    }
  }
}

void dataVeri(void* mem, size_t offset1, size_t offset2, size_t len) {}


std::string randData(size_t offset, size_t len) {
  char key[11] = {0};
  gen_random(key, 10);
  return std::string(key);
}