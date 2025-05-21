#ifndef BALANCE_SHM_H
#define BALANCE_SHM_H

#include <unistd.h>
#include <semaphore.h>

#include <string>
#include <memory>

namespace balance {
namespace util {

class SharedMemory;

class SharedMemory {
public:
    SharedMemory(std::string name, size_t size, bool create = false);
    ~SharedMemory();
    std::string getName();
    size_t getSize();
    char* getPointer();
    void lock();
    void unlock();

private:
    bool _own;

    std::string _name;
    size_t _size;
    int _shmFd;
    char* _shmMemory;
    
    std::string _sem_name;
    sem_t* _shmLock;
};

} //namespace util
} //namespace balance

#endif