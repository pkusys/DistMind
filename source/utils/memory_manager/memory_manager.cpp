#include <unistd.h>

#include <string>
#include <memory>
#include <vector>
#include <iostream>

#include "utils/common/pointer.h"

#include "memory_manager.h"

using namespace std;
using namespace balance::util;

namespace balance {
namespace util {

vector<OffsetPointer> BlockGroup::accessFraction(size_t target_offset, size_t target_size) {
    size_t target_end = target_offset + target_size;
    
    vector<OffsetPointer> fractions;
    if (target_offset < 0 || target_size < 0 || target_end > size)
        return fractions;

    OffsetPointer* cur_block = &data0;
    size_t cur_block_left = 0;
    size_t cur_block_right = cur_block_left + cur_block->size;
    while (cur_block_right <= target_offset) {
        ++cur_block;
        cur_block_left = cur_block_right;
        cur_block_right += cur_block->size;
    }

    while (cur_block_left < target_end) {
        size_t cur_local_offset = max(cur_block_left, target_offset);
        size_t cur_global_offset = cur_block->offset + (cur_local_offset - cur_block_left);
        size_t fraction_size = min(cur_block_right, target_end) - cur_local_offset;
        fractions.push_back(OffsetPointer(cur_global_offset, fraction_size));

        ++cur_block;
        cur_block_left = cur_block_right;
        cur_block_right += cur_block->size;
    }
    return fractions;
}

MemoryManager::MemoryManager(string shm_name, size_t shm_size, size_t block_size):
_block_size(block_size) {
    _shm.reset(new SharedMemory(shm_name, shm_size, true));

    int num_blocks = shm_size / block_size;
    for (int i = 0; i < num_blocks; ++i){
        std::lock_guard<std::mutex> lock(_block_pool_mutex); 
        _block_pool.push(i * block_size);
    }
}

MemoryManager::~MemoryManager() {}

string MemoryManager::getShmName() {
    return _shm->getName();
}

size_t MemoryManager::getShmSize() {
    return _shm->getSize();
}

char* MemoryManager::getShmPointer() {
    return _shm->getPointer();
}

shared_ptr<BlockGroup> MemoryManager::allocate(size_t size) {
    int required_block_num = (size + _block_size - 1) / _block_size;
    int bg_size = sizeof(BlockGroup) + sizeof(OffsetPointer) * (required_block_num - 1);
    cerr << "Allocate: " << size << ", " << required_block_num << "/" << _block_pool.size() << endl;

    shared_ptr<BlockGroup> bg((BlockGroup*)malloc(bg_size));
    bg->num = required_block_num;
    bg->size = size;
    OffsetPointer* ptr = &(bg->data0);
    size_t allocated_size = 0;
    for (ssize_t i = 0; i < bg->num; ++i) {
        while (_block_pool.empty()) {
            cerr << "No more memory to allocate" << endl;
            sleep(600);
        }
        {
            std::lock_guard<std::mutex> lock(_block_pool_mutex); 
            if (_block_pool.empty()){
                --i;
                continue;
            }
            ptr[i].offset = _block_pool.front();
            _block_pool.pop();
        }
        ptr[i].size = min<size_t>(size - allocated_size, _block_size);
        allocated_size += ptr[i].size;
    }
    return bg;
}

void MemoryManager::deallocate(shared_ptr<BlockGroup> bg) {
    if (bg == NULL)
        return;

    cerr << "Deallocate: " << bg->size << ", " << bg->num << "/" << _block_pool.size() << endl;

    OffsetPointer* ptr = &(bg->data0);
    for (size_t i = 0; i < bg->num; ++i){
        std::lock_guard<std::mutex> lock(_block_pool_mutex); 
        _block_pool.push(ptr[i].offset);
    }
}

size_t MemoryManager::getBlockAvailable() {
    return _block_pool.size();
}

size_t MemoryManager::getBlockSize() {
    return _block_size;
}

} //namespace util
} //namespace balance