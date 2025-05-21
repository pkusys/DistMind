#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <semaphore.h>

#include <string>

#include "utils/common/errno.h"

#include "shared_memory.h"

namespace balance {
namespace util {

using namespace std;

const string SEM_SUFFIX = "-mtx";

SharedMemory::SharedMemory(string name, size_t size, bool create):
_own(create), _name(name), _size(size) {
    _shmFd = shm_open(_name.c_str(), O_CREAT | O_RDWR, 0666);
    if (_own)
        ftruncate(_shmFd, _size);
    _shmMemory = (char*)mmap(0, _size, PROT_READ | PROT_WRITE, MAP_SHARED, _shmFd, 0);

    _sem_name = std::string("/" + _name + SEM_SUFFIX);
    _shmLock = sem_open(_sem_name.c_str(), O_CREAT, S_IRUSR | S_IWUSR, 1);
}

SharedMemory::~SharedMemory() {
    munmap(_shmMemory, _size);
    if (_own) {
        shm_unlink(_name.c_str());
        sem_unlink(_sem_name.c_str());
    }
}

std::string SharedMemory::getName() {
    return _name;
}
size_t SharedMemory::getSize() {
    return _size;
}
char* SharedMemory::getPointer() {
    return _shmMemory;
}
void SharedMemory::lock() {
    sem_wait(_shmLock);
}
void SharedMemory::unlock() {
    sem_post(_shmLock);
}

} //namespace util
} //namespace balance