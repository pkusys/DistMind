#ifndef STORE_H
#define STORE_H

#include <atomic>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <condition_variable>

#include "util.h"
#include "efa_ep.h"
#include "shm_common.h"
#include "sock_cli_serv.h"

namespace pipeps {
namespace store {

inline void check_err(bool cond, std::string msg) {
  if (cond)
    std::cerr << msg;
};

// from ps-lite
template <typename T>
class ThdSafeQueue {
 public:
  ThdSafeQueue(){};
  ~ThdSafeQueue(){};

  /**
   * \brief push an value into the end. threadsafe.
   * \param new_value the value
   */
  void push(T new_value) {
    mu_.lock();
    queue_.push(std::move(new_value));
    mu_.unlock();
    cond_.notify_all();
  }

  /**
   * \brief wait until pop an element from the beginning, threadsafe
   * \param value the poped value
   */
  void pop(T* value) {
    std::unique_lock<std::mutex> lk(mu_);
    cond_.wait(lk, [this] { return !queue_.empty(); });
    *value = std::move(queue_.front());
    queue_.pop();
  }

 private:
  mutable std::mutex mu_;
  std::queue<T> queue_;
  std::condition_variable cond_;
};

enum InstrType { PUSH, PULL };
class Instr {
 public:
  int commIdx;  // idx of communicator
  InstrType type;
  std::string key;  // model name or partition id
  int nBatch;       // num of parameter batches for

  // useful when Pushing parameters not previous stored in
  std::vector<size_t> bufs;
  Instr() {
    // set it for sentinel check
    // valid idx must greater than 0;
    commIdx = -1;
  }
};

class ParamStore;
// communicator agent manage multiple communicators
class CommAgent {
 public:
  std::string name;
  ParamStore* store;
  int nComm;
  // params for creating communicator
  int nw;
  std::string data_buf_name;
  size_t data_buf_size;

  std::vector<void*> commInstrPtrs;
  std::vector<sem_t*> commInstrMtxs;
  std::vector<void*> commCntrPtrs;
  std::vector<sem_t*> commCntrMtxs;

  // thread management
  std::vector<std::thread> commThds;
  std::vector<std::thread> commWorkerThds;
  // communicator mangement
  std::vector<trans::shm::SHMCommunicator*> commPtrs;
  CommAgent(int nw,
            std::string agentName,
            std::string data_buf_name,
            size_t data_buf_size,
            ParamStore* s);
  // create communicator and setup peer addrs
  // return the idx of new created communicator
  int getCommunicator(char* peerAddrs, char* commAddrs);

  void cleanComm(int idx);

  // recv data through EFA
  // put recv instr in comms[commIdx]'s shm
  void EFARecv(Instr& ins);

  // put send instr in comms[commIdx]'s shm
  void EFASend(Instr& ins);

  void _setEFAInstr(Instr& ins);

  size_t getCommCntr(int idx);
};

class ParamStore {
 public:
  std::string storeName;
  std::string port;
  std::atomic<bool> _exit{false};
  std::unordered_map<std::string, std::vector<std::pair<size_t, size_t>>>
      memStore;
  std::atomic<size_t> curOffset{
      0};  // the current offset; will be changed by getBuf operation
  ThdSafeQueue<Instr> taskq;
  CommAgent* cAgent;

  ParamStore(std::string name, std::string port, size_t mem_size, int commNw);
  ~ParamStore();
  // get offsets based on the keyname in ins
  void getBufs(Instr& ins, std::vector<std::pair<size_t, size_t>>& bufs);
  // wrapper logics to start threads, initialization environments;
  void run();
};


class StoreCli {
 public:
  std::string name, dstIP, dstPort, nameOfCache;
  int nw;
  size_t cacheSize;

  std::vector<std::thread> wThds;
  ThdSafeQueue<std::pair<int, std::vector<std::pair<size_t, size_t>>>> efaTaskQ;
  std::thread cThd;
  trans::SockCli sCli;
  std::atomic<bool> _exit{false};
  std::thread* efaExcThd;

  // set instr and get status from comm
  void* shmCommInst;
  void* shmCommCntr;
  sem_t* semCommInst;
  sem_t* semCommCntr;

  trans::shm::SHMCommunicator* comm;

  StoreCli(std::string cliName,
           std::string servIP,
           std::string servPort,
           std::string nameOfCache,
           size_t sizeOfCache,
           int nw);
  ~StoreCli();
  // internal init function
  void _init();

  //
  void _open_shm_sem(std::string& commName);

  size_t getCommCntr();

  // push params
  void push(std::string& key, std::vector<std::pair<size_t, size_t>>& dataLoc);

  void pull(std::string& key, std::vector<std::pair<size_t, size_t>>& dataLoc);

};

void cliEFATaskExcThd(StoreCli* cliPtr);

void cliConnHandlerThd(ParamStore* store, int cli);

// this thread can be removed
// this thread is intend to reduce the computation while receiving instructions
void instrHandlerThd(ParamStore* store);


void sockServThd(ParamStore* store, std::string port);

// run communicator inside it
void commThd(trans::shm::SHMCommunicator* comm);
// run worker inside
void workerThd(std::string comm_name,
               int nw,
               int rank,
               std::string data_buf_name,
               size_t data_buf_size);

};  // namespace store
};  // namespace pipeps
#endif
