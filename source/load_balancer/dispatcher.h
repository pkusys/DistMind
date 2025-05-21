#ifndef BALANCE_LB_DISPATCHER_H
#define BALANCE_LB_DISPATCHER_H

#include <unistd.h>

#include <string>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <iostream>

#include "utils/utils.h"

template<class U, class V>
class Dispatcher {
public:
    Dispatcher() {};
    ~Dispatcher() {};

    void registerCustomer(uint64_t id);
    void eraseCustomer(uint64_t id);
    void registerService(uint64_t id);
    void eraseService(uint64_t id);
    void push(std::shared_ptr<U> task);
    std::shared_ptr<U> dispatcherPop();
    void dispatcherPush(uint64_t id, std::shared_ptr<U> task);
    std::shared_ptr<U> serviceFetch(uint64_t id);
    std::shared_ptr<U> serviceFetchBlocked(uint64_t id);
    void serviceComplete(uint64_t id, std::shared_ptr<V> task);
    std::shared_ptr<V> customerPop(uint64_t id);
    std::shared_ptr<V> customerPopBlocked(uint64_t id);

private:
    balance::util::AtomicQueue<std::shared_ptr<U> > _q_in;
    std::unordered_map<uint64_t, balance::util::AtomicQueue<std::shared_ptr<U> > > _q_working;
    std::unordered_map<uint64_t, balance::util::AtomicQueue<std::shared_ptr<V> > > _q_out;
    std::mutex _lock_in;
    std::mutex _lock_working;
    std::mutex _lock_out;
};

template<class U, class V>
void Dispatcher<U, V>::registerCustomer(uint64_t id) {
    const std::lock_guard<std::mutex> guard(_lock_out);
    _q_out[id] = balance::util::AtomicQueue<std::shared_ptr<V> >();
}

template<class U, class V>
void Dispatcher<U, V>::eraseCustomer(uint64_t id) {
    const std::lock_guard<std::mutex> guard(_lock_out);
    _q_out.erase(id);
}

template<class U, class V>
void Dispatcher<U, V>::registerService(uint64_t id) {
    const std::lock_guard<std::mutex> guard(_lock_working);
    _q_working[id] = balance::util::AtomicQueue<std::shared_ptr<U> >();
}

template<class U, class V>
void Dispatcher<U, V>::eraseService(uint64_t id) {
    const std::lock_guard<std::mutex> guard(_lock_working);
    _q_working.erase(id);
}

template<class U, class V>
void Dispatcher<U, V>::push(std::shared_ptr<U> task) {
    const std::lock_guard<std::mutex> guard(_lock_in);
    _q_in.push(task);
}

template<class U, class V>
std::shared_ptr<U> Dispatcher<U, V>::dispatcherPop() {
    const std::lock_guard<std::mutex> guard(_lock_in);
    
    if (!_q_in.empty())
        return _q_in.pop();
    
    return nullptr;
}

template<class U, class V>
void Dispatcher<U, V>::dispatcherPush(uint64_t id, std::shared_ptr<U> task) {
    const std::lock_guard<std::mutex> guard(_lock_working);
    _q_working[id].push(task);
}

template<class U, class V>
std::shared_ptr<U> Dispatcher<U, V>::serviceFetch(uint64_t id) {
    const std::lock_guard<std::mutex> guard(_lock_working);
    if (_q_working[id].empty())
        return nullptr;
    return _q_working[id].pop();
}

template<class U, class V>
std::shared_ptr<U> Dispatcher<U, V>::serviceFetchBlocked(uint64_t id) {
    while (true) {
        auto ret = serviceFetch(id);
        if (ret != nullptr)
            return ret;
        usleep(10);
    }
}

template<class U, class V>
void Dispatcher<U, V>::serviceComplete(uint64_t id, std::shared_ptr<V> task) {
    const std::lock_guard<std::mutex> guard(_lock_out);
    _q_out[id].push(task);
}

template<class U, class V>
std::shared_ptr<V> Dispatcher<U, V>::customerPop(uint64_t id) {
    const std::lock_guard<std::mutex> guard(_lock_out);
    if (_q_out[id].empty())
        return nullptr;
    return _q_out[id].pop();
}

template<class U, class V>
std::shared_ptr<V> Dispatcher<U, V>::customerPopBlocked(uint64_t id) {
    while (true) {
        auto ret = customerPop(id);
        if (ret != nullptr)
            return ret;
        usleep(10);
    }
}

#endif