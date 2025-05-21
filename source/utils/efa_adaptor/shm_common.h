#ifndef SHM_COMMON_H
#define SHM_COMMON_H

#include <fcntl.h>
#include <semaphore.h>
#include <sys/mman.h>
#include <sys/stat.h> /* For mode constants */
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>

#include "efa_ep.h"
#include "util.h"
#include <rdma/fi_tagged.h>

namespace trans {
namespace shm {
void check_err(bool cond, std::string msg);

bool check_all_zero(char* buf, size_t size);

const int INSTR_OFFSET = 12;
const int INSTR_SIZE = 10 * 1024;
const int INSTR_DATA_SIZE = INSTR_SIZE - INSTR_OFFSET;
const int EFA_ADDR_SIZE = 64;
const int CNTR_SIZE = 8;
const int STATUS_SIZE = 4;

const std::string SHM_SUFFIX_INSTR = "-instr-mem";
const std::string SHM_SUFFIX_CNTR = "-cntr-mem";
const std::string SHM_SUFFIX_EFA_ADDR = "-efa-addr-mem";
const std::string SHM_SUFFIX_W_STAT = "-worker-status-mem";
const std::string SHM_SUFFIX_DATA_BUF = "-data-buf-mem";

const std::string SEM_SUFFIX_INSTR = "-instr-mtx-";
const std::string SEM_SUFFIX_CNTR = "-cntr-mtx-";
const std::string SEM_SUFFIX_EFA_ADDR = "-efa-addr-mtx-";
const std::string SEM_SUFFIX_W_STAT = "-worker-status-mtx-";
const std::string SEM_SUFFIX_DATA_BUF = "-data-buf-mtx-";

void shm_lock(sem_t* s, std::string msg_if_err);

void shm_unlock(sem_t* s, std::string msg_if_err);

void print_sem_mutex_val(sem_t* s);

int sem_mutex_val(sem_t* s);

enum INSTR_T { ERR_INSTR, SET_EFA_ADDR, RECV_BATCH, SEND_BATCH, SHUTDOWN};

class Instruction {
 public:
  INSTR_T type;
  double timestamp;
  char data[INSTR_DATA_SIZE] = {0};
  Instruction(){};
};

INSTR_T instr_map(int idx);

int reverse_map(INSTR_T t);

class WorkerMemory {
 public:
  // 8 bytes for timestamp;
  // 4 bytes for instr-code; the remaining bytes for instr data
  int instr_size = INSTR_SIZE;
  int efa_addr_size = EFA_ADDR_SIZE;
  int cntr_size = CNTR_SIZE;             // 4 Bytes
  int status_size = STATUS_SIZE;         // indicate the status of worker
  unsigned long long int data_buf_size;  // input from user
  // based on rank to move the pointer of instr, efa_addr, cntr, status
  int rank;
  int nw;  // number of workers

  // shm identifiers
  std::string shm_instr;
  std::string shm_cntr;
  std::string shm_data_buf;
  std::string shm_efa_addr;
  std::string shm_w_status;

  // mutex names
  std::string sem_name_instr;
  std::string sem_name_cntr;
  std::string sem_name_data;
  std::string sem_name_efa_addr;
  std::string sem_name_w_status;

  // mutex pointers
  sem_t* sem_instr;
  sem_t* sem_cntr;
  sem_t* sem_data;
  sem_t* sem_efa_addr;
  sem_t* sem_w_status;

  // pointers
  void* instr_ptr;
  void* cntr_ptr;
  void* data_buf_ptr;
  void* efa_add_ptr;
  void* status_ptr;  // 1: idle; 2: working;

  /* rank start from 0
    nw: total workers
   */
  WorkerMemory(std::string prefix,
               int nw,
               int rank,
               std::string data_buf_name,
               size_t data_size);

  void open_shm_sem();

  ~WorkerMemory();
};

class SHMWorker {
  std::string comm_name;
  int rank;
  EFAEndpoint* efa_ep;
  WorkerMemory* mem;

 public:
  SHMWorker(std::string comm_name,
            int nw,
            int rank,
            std::string data_buf_name,
            size_t shared_data_size);

  Instruction* read_instr();

  void set_local_efa_addr(EFAEndpoint* efa);

  void _wait_cq(fid_cq* cq, int count);

  /* insert remote efa addr into address vector */
  void set_remote_efa_addr(EFAEndpoint* efa, Instruction* i);

  void efa_send_recv_batch(EFAEndpoint* efa, Instruction* instr);

  void shutdown();

  // main function of the shm worker
  void run();

  ~SHMWorker();
};

class SHMCommunicator {
 public:
  int nw;
  std::string name;
  std::string data_buf_name;
  size_t data_buf_size;
  // shm for workers
  std::string ws_instr_shm_name;
  std::string ws_cntr_shm_name;
  std::string ws_efa_addr_shm_name;
  std::string ws_status_shm_name;

  void* ws_instr_ptr;
  void* ws_cntr_ptr;
  void* ws_efa_add_ptr;
  void* ws_status_ptr;
  void* data_buf_ptr;

  // mutex
  std::vector<sem_t*> mtxs_instr;
  std::vector<sem_t*> mtxs_cntr;
  std::vector<sem_t*> mtxs_efa_addr;
  std::vector<sem_t*> mtxs_w_status;
  std::vector<sem_t*> mtxs_w_data_buf;

  // instr memory for communicator
  std::string shm_comm_instr;
  // cntr memory for communicator
  std::string shm_comm_cntr;
  // comm shm ptrs
  void* comm_instr_ptr;
  void* comm_cntr_ptr;
  // comm sem mtx
  sem_t* mtx_comm_instr;
  sem_t* mtx_comm_cntr;

  SHMCommunicator(int num_workers,
                  std::string name,
                  std::string data_buf_name,
                  size_t data_buf_size);

  void create_workers_shm_sem();

  /* create shared memory for upper level API calls */
  void create_self_shm_sem();

  bool local_efa_addrs_ready();

  /* check ready status first */
  void get_local_efa_addrs(char* addrs_buf);

  /* the addrs_buf contains addresses for all workers */
  void set_local_peer_addrs(char* addrs_buf);

  void get_workers_cntr(size_t* cntrs);

  int get_a_worker_cntr(int widx);

  void plus_one_self_cntr();

  /* offsets are relative to the data_buf_ptr */
  void send_recv_batch(Instruction* instr, bool round_robin = true);

  Instruction* _read_instr();

  // shutdown command of all workers
  void closeComm();

  void run();

  ~SHMCommunicator();
};

};  // namespace shm
};  // namespace trans

#endif