#include <fstream>
#include <execinfo.h>
#include <signal.h>

#include "utils/utils.h"

void seghandler(int sig) {
  void *array[10];
  size_t size;

  // get void*'s for all entries on the stack
  size = backtrace(array, 10);

  // print out all the frames to stderr
  fprintf(stderr, "Error: signal %d:\n", sig);
  backtrace_symbols_fd(array, size, STDERR_FILENO);
  exit(1);
}

using namespace pipeps::store;

void* openSHM(std::string& shmName, size_t& shmSize);

size_t loadParamToSHM(void* memPtr,
                      std::vector<std::pair<size_t, size_t>>& dataLoc);

void getRecvLoc(size_t& curOffset,
                std::vector<std::pair<size_t, size_t>>& sendFrom,
                std::vector<std::pair<size_t, size_t>>& recvTo);

bool verifyData(char* memPtr,
                std::vector<std::pair<size_t, size_t>>& sendFrom,
                std::vector<std::pair<size_t, size_t>>& recvTo);

void clearData(void* memPtr, std::vector<std::pair<size_t, size_t>>& d);

int main(int argc, char const* argv[]) {
  signal(SIGSEGV, seghandler);
  if (argc < 6) {
    std::cout
        << "Usage: ./store_cli <dstIp> <dstPort> <cacheName> <cacheSize> <cliName>\n";
    return -1;
  }
  std::string ip(argv[1]);
  std::string port(argv[2]);
  std::string cacheName(argv[3]);
  size_t cacheSize = std::stoull(argv[4]);
  std::string cname(argv[5]);

  std::string modelKey = "resnet152-test";
  
  void* dataPtr = openSHM(cacheName, cacheSize);

  std::unique_ptr<StoreCli> sCli(
      new StoreCli(cname, ip, port, cacheName, cacheSize, 4));
  // make sure shm opened
  std::this_thread::sleep_for(std::chrono::seconds(1));
  
  std::vector<std::pair<size_t, size_t>> sendDataLoc;
  std::vector<std::pair<size_t, size_t>> recvDataLoc;
  size_t cacheOffset = loadParamToSHM(dataPtr, sendDataLoc);
  if (cacheOffset == 0) {
    std::cerr << "error happend while loading parameters\n";
    return -1;
  }
  size_t paramSize = cacheOffset;
  getRecvLoc(cacheOffset, sendDataLoc, recvDataLoc);
  std::cout << "params memory size: " + std::to_string(paramSize) + "\n";
  int curCntr = sCli->getCommCntr();
  int target = curCntr + sendDataLoc.size();
  double s = trans::time_now();
  sCli->push(modelKey, sendDataLoc);
  while (sCli->getCommCntr() != target) {
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  std::cout << "push parameters completed"
            << std::endl << std::endl << std::endl;
  sleep(5);

  // receive multiple times
  for (int i = 0; i < 5; i++) {
    double s = trans::time_now();
    curCntr = sCli->getCommCntr();
    target = curCntr + recvDataLoc.size();
    sCli->pull(modelKey, recvDataLoc);
    while (sCli->getCommCntr() != target) {
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    double e = trans::time_now();
    bool v = verifyData((char*)dataPtr, sendDataLoc, recvDataLoc);
    if (v)
      std::cout << "data correct\n";
    else
      std::cout << "received data wrong\n";

    double dur = e - s;
    double bw = ((paramSize * 8) / dur) / 1e9;
    std::cout << "bw: " << bw << " Gbps; "
              << " dur: " << dur
              << std::endl << std::endl << std::endl;
    sleep(5);
    // clean received;
    clearData(dataPtr, recvDataLoc);
  }

  // explicitly unlink shm
  shm_unlink(cacheName.c_str());
  return 0;
}

void* openSHM(std::string& shmName, size_t& shmSize) {
  int data_fd = shm_open(shmName.c_str(), O_CREAT | O_RDWR, 0666);
  check_err((ftruncate(data_fd, shmSize) < 0), "ftruncate instr_fd err\n");
  void* data_buf_ptr =
      mmap(0, shmSize, PROT_READ | PROT_WRITE, MAP_SHARED, data_fd, 0);
  return data_buf_ptr;
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

size_t loadParamToSHM(void* memPtr,
                      std::vector<std::pair<size_t, size_t>>& dataLoc) {
  size_t _offset = 0;
  for (int i = 0; i < 10; ++i) {
    char* buf = (char*)memPtr + _offset;
    size_t len = 20UL * 1024UL * 1024UL;
    for (int i = 0; i < len / sizeof(double); ++i)
      ((double*)buf)[i] = (double)i;
    std::cout << "load data size: " + std::to_string(len) << "\n";
    dataLoc.push_back(std::make_pair(_offset, len));
    _offset += len;
  }
  return _offset;
}

void getRecvLoc(size_t& curOffset,
                std::vector<std::pair<size_t, size_t>>& sendFrom,
                std::vector<std::pair<size_t, size_t>>& recvTo) {
  for (auto p : sendFrom) {
    recvTo.push_back(std::make_pair(p.first + curOffset, p.second));
  }
}

bool verifyData(char* memPtr,
                std::vector<std::pair<size_t, size_t>>& sendFrom,
                std::vector<std::pair<size_t, size_t>>& recvTo) {
  bool same = true;
  for (int i = 0; i < sendFrom.size(); i++) {
    auto _s = sendFrom[i];
    auto _r = recvTo[i];
    for (int j = 0; j < _s.second; j++) {
      if (*(memPtr + _s.first + j) != *(memPtr + _r.first + j)) {
        same = false;
        return same;
      } else {
        if (j < 10) {
          std::cout << *((char*)memPtr + _r.first + j);
        }
      }
    }
  }
  std::cout << "\n";
  return same;
}

void clearData(void* memPtr, std::vector<std::pair<size_t, size_t>>& d) {
  for (auto b : d) {
    std::fill_n((char*)memPtr + b.first, b.second, 0);
  }
}