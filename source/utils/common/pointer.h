#ifndef BALANCE_POINTER_H
#define BALANCE_POINTER_H

#include <unistd.h>

#include <memory>

namespace balance {
namespace util {

struct AddressPointer {
    char* ptr;
    size_t size;

    AddressPointer(char* _ptr, size_t _size):
    ptr(_ptr), size(_size) {}
    AddressPointer(const AddressPointer &ap):
    ptr(ap.ptr), size(ap.size) {}
};

struct OffsetPointer {
    size_t offset;
    size_t size;

    OffsetPointer(size_t _offset, size_t _size):
    offset(_offset), size(_size) {}
    OffsetPointer(const OffsetPointer &op):
    offset(op.offset), size(op.size) {}
};

} //namespace util
} //namespace balance

#endif