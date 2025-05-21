#ifndef BALANCE_TIME_H
#define BALANCE_TIME_H

#include <chrono>

namespace balance {
namespace util {

inline double time_now() {
    auto t = std::chrono::high_resolution_clock::now();
    return t.time_since_epoch().count() / 1e9; // convert to seconds
};

} //namespace util
} //namespace balance

#endif