#include <atomic>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <condition_variable>
#include <sstream>

#include "efa_adaptor.h"

namespace pipeps {
namespace store {

StoreCli::StoreCli(std::string cliName,
                   std::string servIP,
                   std::string servPort,
                   std::string nameOfCache,
                   size_t sizeOfCache,
                   int nw) {
  std::cout << cliName << ", " << servIP << ", " << servPort << ", " << nameOfCache << ", " << sizeOfCache << std::endl;
  this->name = cliName;
  this->dstIP = servIP;
  this->dstPort = servPort;
  this->nameOfCache = nameOfCache;
  cacheSize = sizeOfCache;
  this->nw = nw;  // num of workers for communicator

  this->_init();
}

StoreCli::~StoreCli(){
  this->_exit = true;
  trans::shm::shm_lock(semCommInst,
                       "cleanComm put inst: lock err");
  // set the Instruction shutdown
  void* _instr_ptr = shmCommInst;
  *(int*)((char*)_instr_ptr + 8) = trans::shm::reverse_map(trans::shm::SHUTDOWN);
  *(double*)_instr_ptr = trans::time_now();

  trans::shm::shm_unlock(semCommInst,
                         "cleanComm put inst: unlock err");
  // 
  while (true){
    int _c = getCommCntr();
    std::cout << "communicator cntr (int) " << _c << "\n"; 
    if (_c < 0) {
      break;
    } else {
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
  }
  // put empty task for efaTaskQ to quit for waiting
  efaTaskQ.push(std::make_pair(-1, std::vector<std::pair<size_t, size_t>>()));
  if (this->efaExcThd->joinable()) {
    this->efaExcThd->join();
  }
  std::cout << "exiting wait for workers \n";
  for (int i = 0; i < wThds.size(); i++) {
    if (wThds[i].joinable()) {
      wThds[i].join();
    }
  }
  std::cout << "exiting wait for communication thd\n";
  if (cThd.joinable()) {
    cThd.join();
  }
  // delete newed object
  delete comm;
  delete this->efaExcThd;
}

void cliEFATaskExcThd(StoreCli* cliPtr){
  void* shmCommInst = cliPtr->shmCommInst;

  while (!cliPtr->_exit) {
    std::pair<int, std::vector<std::pair<size_t, size_t>>> t;
    cliPtr->efaTaskQ.pop(&t);
    if (cliPtr->_exit) {break;} // at last will receive a empty element
    int ops = t.first;
    size_t curCntr = cliPtr->getCommCntr();
    trans::shm::shm_lock(cliPtr->semCommInst, "cliEFATaskExcThd lock err");
    *(int*)((char*)shmCommInst + 8) = ops;
    *(int*)((char*)shmCommInst + 12) = t.second.size();
    char* _batch_data_s = (char*)shmCommInst + 16;
    for (int i = 0; i < t.second.size(); i++) {
      *(size_t*)(_batch_data_s + i * 16) = t.second[i].first;
      *(size_t*)(_batch_data_s + i * 16 + 8) = t.second[i].second;
    }
    *(double*)shmCommInst = trans::time_now();
    trans::shm::shm_unlock(cliPtr->semCommInst, "cliEFATaskExcThd unlock err");

    // wait for task completion
    size_t targetCntr = curCntr + t.second.size();
    while (cliPtr->getCommCntr() != targetCntr) {
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
  }
  std::stringstream ss;
  ss << cliPtr->name << "cliEFATaskExcThd exiting \n";
  std::cout << ss.str();
}

void StoreCli::_init() {
  // create communicator
  std::string commName = name + "-storecli-comm";
  comm = new trans::shm::SHMCommunicator(nw, commName, nameOfCache, cacheSize);
  // create workers of communicators
  for (int i = 0; i < nw; i++) {
    std::thread wt(workerThd, commName, nw, i, nameOfCache, cacheSize);
    wThds.push_back(std::move(wt));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  // wait for workers ready to get EFA address
  while (!comm->local_efa_addrs_ready()) {
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  // EFA addrs exchange
  size_t addrs_size = nw * trans::shm::EFA_ADDR_SIZE;
  char* localAddrs = new char[addrs_size];
  char* peerAddrs = new char[addrs_size];
  comm->get_local_efa_addrs(localAddrs);

  sCli = trans::SockCli(dstIP, dstPort);
  sCli._send(localAddrs, addrs_size);
  sCli._recv(peerAddrs, addrs_size);
  comm->set_local_peer_addrs(peerAddrs);

  std::thread _ct(commThd, comm);
  // _ct.detach();
  cThd = std::move(_ct);

  // init communicator memory
  _open_shm_sem(commName);

  // create thread for execute efa operations one by one
  this->efaExcThd = new std::thread(cliEFATaskExcThd, this);
}

void StoreCli::_open_shm_sem(std::string& commName) {
  std::string _instr_shm_name = commName + "-comm-instr-mem";
  std::string _cntr_shm_name = commName + "-comm-cntr-mem";
  int instr_fd = shm_open(_instr_shm_name.c_str(), O_RDWR, 0666);
  int cntr_fd = shm_open(_cntr_shm_name.c_str(), O_RDWR, 0666);

  this->shmCommInst = mmap(0, trans::shm::INSTR_SIZE, PROT_READ | PROT_WRITE,
                           MAP_SHARED, instr_fd, 0);
  this->shmCommCntr = mmap(0, trans::shm::CNTR_SIZE, PROT_READ | PROT_WRITE,
                           MAP_SHARED, cntr_fd, 0);
  std::string _sem_comm_instr("/" + commName + "-comm-instr-mtx");
  std::string _sem_comm_cntr("/" + commName + "-comm-cntr-mtx");

  this->semCommInst = sem_open(_sem_comm_instr.c_str(), 0);
  this->semCommCntr = sem_open(_sem_comm_cntr.c_str(), 0);
}


void StoreCli::push(std::string& key,
                    std::vector<std::pair<size_t, size_t>>& srcDataLoc) {
  // =============== send ctl msg via TCP =================
  // send instr type
  char type[4];
  *(int*)type = 0;
  sCli._send(type, 4);

  // send key length
  char keylen[4];
  *(int*)keylen = key.size();
  sCli._send(keylen, 4);

  // send key
  // auto keybuf = std::make_unique<char>(key.size()); c++14
  // std::unique_ptr<char> keybuf(new char[key.size()]);
  char* keybuf = new char[key.size()];
  memcpy(keybuf, key.c_str(), key.size());
  sCli._send(keybuf, key.size());
  
  // send nBatch
  char nb[4];
  *(int*)nb = srcDataLoc.size();
  sCli._send(nb, 4);

  // send trunk sizes of data
  for (int i = 0; i < srcDataLoc.size(); i++) {
    char _bs[8];
    *(size_t*)_bs = srcDataLoc[i].second;
    sCli._send(_bs, 8);
  }
  // =============== end ctl msg via TCP =================

  // =============== add task to Info local EFA
  int ops = trans::shm::reverse_map(trans::shm::SEND_BATCH);
  efaTaskQ.push(std::make_pair(ops, srcDataLoc));
  // =============== End EFA instruction
  delete[] keybuf;
}

void StoreCli::pull(std::string& key,
                    std::vector<std::pair<size_t, size_t>>& dataLoc) {
  // =============== send ctl msg via TCP =================
  // send instr type
  char type[4];
  *(int*)type = 1; // read the type: 0 == PUSH; > 0 PULL
  sCli._send(type, 4);

  // send key length
  char keylen[4];
  *(int*)keylen = key.size();
  sCli._send(keylen, 4);

  // send key
  // auto keybuf = std::make_unique<char>(key.size());
  // std::unique_ptr<char> keybuf(new char[key.size()]);
  char* keybuf = new char[key.size()];
  memcpy(keybuf, key.c_str(), key.size());
  sCli._send(keybuf, key.size());

  // send nBatch = 0
  std::stringstream ss;
  ss << "**** storeCli pull: dataLoc #: " << std::to_string(dataLoc.size()) << " Sizes: ";
  char nb[4];
  *(int*)nb = dataLoc.size();
  sCli._send(nb, 4);
  // send trunk sizes of data
  // storage will split data into trunks
  for (int i = 0; i < dataLoc.size(); i++) {
    char _bs[8];
    *(size_t*)_bs = dataLoc[i].second;
    sCli._send(_bs, 8);
    // ss << std::to_string(dataLoc[i].second) << ", ";
  }
  ss << "\n";
  // std::cout << ss.str();
  // =============== end ctl msg via TCP =================

  // =============== Info local EFA
  int ops = trans::shm::reverse_map(trans::shm::RECV_BATCH);
  efaTaskQ.push(std::make_pair(ops, dataLoc));
  // =============== End EFA instruction
  delete[] keybuf;
}

size_t StoreCli::getCommCntr() {
  trans::shm::shm_lock(semCommCntr, "StoreCli::getCommCntr: lock err");
  size_t _c = *(size_t*)shmCommCntr;
  trans::shm::shm_unlock(semCommCntr, "StoreCli::getCommCntr: unlock err");
  return _c;
}

void cliConnHandlerThd(ParamStore* store, int cli) {
  // read the remote efa address from remote
  // assume addrs takes 64 bytes;
  size_t addr_size = store->cAgent->nw * 64;
  char* peer_addrs = new char[addr_size];
  char* local_addrs = new char[addr_size];

  // received from remote
  size_t ret = read(cli, peer_addrs, addr_size);
  check_err((ret != addr_size), "socket read size not match\n");
  int commIdx = store->cAgent->getCommunicator(peer_addrs, local_addrs);
  // make sure communicator started
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  // send to remote
  ret = send(cli, local_addrs, addr_size, 0);
  check_err((ret != addr_size), "socket send err\n");

  int sockErr = 0;
  socklen_t errSize = sizeof(sockErr);
  // handle other instructions
  while (!store->_exit && getsockopt(cli, SOL_SOCKET, SO_ERROR, &sockErr, &errSize) >= 0) {
    Instr ins;
    ins.commIdx = commIdx;
    // read the type: 0 == PUSH; > 0 PULL
    char type[4];
    ret = read(cli, type, 4);
    if (ret != 4) {
      std::cerr << "cli " + std::to_string(cli) << " connection broken\n";
      break; 
    }
    ins.type = *(int*)type > 0 ? PULL : PUSH;
    // read len for key
    char keylen[4];
    read(cli, keylen, 4);
    // read key
    size_t _len = *(int*)keylen;
    char* key = new char[_len];
    read(cli, key, _len);
    ins.key = std::string(key, _len);
    delete[] key;
    // read nbatch
    char nb[4];
    read(cli, nb, 4);
    ins.nBatch = *(int*)nb;
    for (int i = 0; i < ins.nBatch; i++) {
      // if nb is greater than 0
      char s[8];
      read(cli, s, 8);
      ins.bufs.push_back(*(size_t*)s);
    }
    store->taskq.push(ins);
    std::string _msg("Got a request from cli " + std::to_string(cli) +
                       " for comm " + std::to_string(commIdx) + " with type " +
                       std::to_string(ins.type) + ": " + ins.key);
    std::cout << _msg << std::endl;
  }
  //
  store->cAgent->cleanComm(commIdx);
};

// this thread can be removed
// this thread is intend to reduce the computation while receiving instructions
void instrHandlerThd(ParamStore* store) {
  std::cout << "ParamStore instrHandlerThd started\n";
  while (!store->_exit) {
    Instr* ins = new Instr();
    store->taskq.pop(ins);
    if (ins->type == PUSH) {
      store->cAgent->EFARecv(*ins);
    } else if (ins->type == PULL) {
      std::cout << "Push task: " << ins->key << std::endl;
      store->cAgent->EFASend(*ins);
    }

    delete ins;
  }
};

void sockServThd(ParamStore* store, std::string port) {
  // always listen for new connections
  trans::SockServ serv(port);
  std::vector<std::thread> handles;
  while (true && !store->_exit) {
    int sockfd = serv._listen();  // get new connection
    std::thread _t(cliConnHandlerThd, store, sockfd);
    _t.detach();
    handles.push_back(std::move(_t));
  }
  for (int i = 0; i < handles.size(); i++) {
    if (handles[i].joinable())
      handles[i].join();
  }
};

// run communicator inside it
void commThd(trans::shm::SHMCommunicator* comm) {
  comm->run();
  std::cout << "ending of storeCli communication thd\n";
};
// run worker inside
void workerThd(std::string comm_name,
               int nw,
               int rank,
               std::string data_buf_name,
               size_t data_buf_size) {
  trans::shm::SHMWorker w(comm_name, nw, rank, data_buf_name, data_buf_size);
  w.run();
  std::stringstream ss;
  ss << "ending worker thd of " << comm_name << " rank " << rank << "\n";
  std::cout << ss.str();
};

// Arg nw: number of workers for each communicator
CommAgent::CommAgent(int nw,
                     std::string agentName,
                     std::string data_buf_name,
                     size_t data_buf_size,
                     ParamStore* s) {
  this->nw = nw;
  this->nComm = 0;
  this->data_buf_name = data_buf_name;
  this->data_buf_size = data_buf_size;
  this->store = s;
  this->name = agentName;
};

int CommAgent::getCommunicator(char* peerAddrs, char* commAddrs) {
  int idx = nComm;
  // prepare hyper parameters
  std::string commName = this->name + "-comm-" + std::to_string(idx);
  // create communicator
  trans::shm::SHMCommunicator* comm = new trans::shm::SHMCommunicator(
      nw, commName, data_buf_name, data_buf_size);

  // create workers
  for (int i = 0; i < nw; i++) {
    std::thread wt(workerThd, commName, nw, i, data_buf_name, data_buf_size);
    wt.detach();
    commWorkerThds.push_back(std::move(wt));
  }

  // wait for local workers ready to push their EFA addrs
  while (!comm->local_efa_addrs_ready()) {
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }

  comm->set_local_peer_addrs(peerAddrs);
  comm->get_local_efa_addrs(commAddrs);

  this->commPtrs.push_back(comm);
  std::thread ct(commThd, comm);
  ct.detach();
  this->commThds.push_back(std::move(ct));
  // done: get comm Instr, cntr shm
  std::string _instr_shm_name = commName + "-comm-instr-mem";
  std::string _cntr_shm_name = commName + "-comm-cntr-mem";
  int instr_fd = shm_open(_instr_shm_name.c_str(), O_RDWR, 0666);
  int cntr_fd = shm_open(_cntr_shm_name.c_str(), O_RDWR, 0666);

  void* _instr_ptr = mmap(0, trans::shm::INSTR_SIZE, PROT_READ | PROT_WRITE,
                          MAP_SHARED, instr_fd, 0);
  void* _cntr_ptr = mmap(0, trans::shm::CNTR_SIZE, PROT_READ | PROT_WRITE,
                         MAP_SHARED, cntr_fd, 0);
  commInstrPtrs.push_back(_instr_ptr);
  commCntrPtrs.push_back(_cntr_ptr);

  std::string _sem_comm_instr("/" + commName + "-comm-instr-mtx");
  std::string _sem_comm_cntr("/" + commName + "-comm-cntr-mtx");

  sem_t* _instr_mtx = sem_open(_sem_comm_instr.c_str(), 0);
  sem_t* _cntr_mtx = sem_open(_sem_comm_cntr.c_str(), 0);
  commInstrMtxs.push_back(_instr_mtx);
  commCntrMtxs.push_back(_cntr_mtx);

  nComm++;
  return idx;
}

size_t CommAgent::getCommCntr(int idx) {
  void* cntrPtr = commCntrPtrs[idx];
  sem_t* cntrMtx = commCntrMtxs[idx];

  trans::shm::shm_lock(cntrMtx, "CommAgent::getCommCntr lock cntr");
  size_t cntr = *(size_t*) cntrPtr;
  trans::shm::shm_unlock(cntrMtx, "CommAgent::getCommCntr unlock cntr err");

  return cntr;
}

void CommAgent::cleanComm(int idx){
  trans::shm::shm_lock(commInstrMtxs[idx],
                       "cleanComm put inst: lock err");
  // set the Instruction shutdown
  void* _instr_ptr = commInstrPtrs[idx];
  *(int*)((char*)_instr_ptr + 8) = trans::shm::reverse_map(trans::shm::SHUTDOWN);
  *(double*)_instr_ptr = trans::time_now();

  trans::shm::shm_unlock(commInstrMtxs[idx],
                         "cleanComm put inst: unlock err");
  // 
  while (true){
    trans::shm::shm_lock(commCntrMtxs[idx], "StoreCli::getCommCntr: lock err");
    int _c = *(int*)commCntrPtrs[idx];
    trans::shm::shm_unlock(commCntrMtxs[idx], "StoreCli::getCommCntr: unlock err");
    if (_c < 0) {
      std::cout << "clean while communicator cntr: " << _c << "\n";
      break;
    }
  }
  // delete the communicator to shm_unlink
  delete commPtrs[idx];
}

void CommAgent::_setEFAInstr(Instr& ins) {
  int ops;  // operation code
  if (ins.type == PUSH) {
    ops = trans::shm::reverse_map(trans::shm::RECV_BATCH);
  } else {
    ops = trans::shm::reverse_map(trans::shm::SEND_BATCH);
  }
  // get buf
  std::vector<std::pair<size_t, size_t>> bufs;
  this->store->getBufs(ins, bufs);
  if (ins.type == PUSH) {
    // record the memory location for key
    this->store->memStore[ins.key] = bufs;
  }
  std::cout << "CommAgent::_setEFAInstr bufs size: " + std::to_string(bufs.size()) + "\n";
  size_t curCntr = getCommCntr(ins.commIdx);
  // write instruction to corresponding communicator mem
  trans::shm::shm_lock(commInstrMtxs[ins.commIdx],
                       "_setEFAInstr put inst: lock err");
  void* _instr_ptr = commInstrPtrs[ins.commIdx];
  *(int*)((char*)_instr_ptr + 8) = ops;
  *(int*)((char*)_instr_ptr + 12) = bufs.size();
  char* _batch_data_s = (char*)_instr_ptr + 16;
  for (int i = 0; i < bufs.size(); i++) {
    *(size_t*)(_batch_data_s + i * 16) = bufs[i].first;
    *(size_t*)(_batch_data_s + i * 16 + 8) = bufs[i].second;
  }
  *(double*)_instr_ptr = trans::time_now();
  trans::shm::shm_unlock(commInstrMtxs[ins.commIdx],
                         "_setEFAInstr put inst: unlock err");
  // wait for communicators to complete
  // sync
  size_t targetCntr = curCntr + bufs.size();
  while (this->getCommCntr(ins.commIdx) != targetCntr) {
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
}

void CommAgent::EFARecv(Instr& ins) {
  _setEFAInstr(ins);
}

void CommAgent::EFASend(Instr& ins) {
  std::cout << "request key: " + ins.key +  "; Send params EFA\n";
  _setEFAInstr(ins);
}

ParamStore::ParamStore(std::string name,
                       std::string port,
                       size_t mem_size,
                       int commNw) {
  this->storeName = name;
  this->port = port;

  // hyper parameter
  std::string data_buf_name = name + "data-buf-mem";

  int data_buf_fd = shm_open(data_buf_name.c_str(), O_CREAT | O_RDWR, 0666);
  check_err((ftruncate(data_buf_fd, mem_size) < 0), "ftruncate instr_fd err\n");
  void* data_buf_ptr = mmap(0, mem_size, PROT_READ | PROT_WRITE, MAP_SHARED,
                      data_buf_fd, 0);

  cAgent = new CommAgent(commNw, name + "-cAgent", data_buf_name, mem_size, this);
}

ParamStore::~ParamStore() {
}

void ParamStore::getBufs(Instr& ins,
                         std::vector<std::pair<size_t, size_t>>& bufs) {
                          
  // if the key is not exist; need to move the cur pointer
  
  auto _it = memStore.find(ins.key);
  if (_it != memStore.end()) {
    std::cout << "ParamStore::getBufs key: " + ins.key + " exists\n";
    // bufs = _it->second;
    // using the partion from the ins, but the first offset the same
    size_t bufStart = _it->second[0].first;
    for (size_t& bs : ins.bufs) {
      bufs.push_back(std::make_pair(bufStart, bs));
      bufStart += bs;
    }
  } else {
    // key is not exist
    std::string _msg("ParamStore::getBufs key: " + ins.key + " not exist\n");
    std::cout << _msg;
    for (size_t& bs : ins.bufs) {
      std::string _msg("ParamStore::getBufs curOffset: " + std::to_string(curOffset) + "; Buf size " + std::to_string(bs) + "\n");
      std::cout << _msg;
      bufs.push_back(std::pair<size_t, size_t>(curOffset, bs));
      curOffset += bs;
    }
  }
};

void ParamStore::run() {
  std::thread sockServ(sockServThd, this, port);
  std::thread instrHandle(instrHandlerThd, this);

  sockServ.join();
  instrHandle.join();
}

};  // namespace store
};  // namespace pipeps