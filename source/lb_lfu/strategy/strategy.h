#ifndef BALANCE_LB_STRATEGY_TYPES_H
#define BALANCE_LB_STRATEGY_TYPES_H

#include <unistd.h>

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>

#include "../metadata_manager.h"

struct ServerSelection {
    int ip_int;
    int port;

    ServerSelection(int _ip_int, int _port):
    ip_int(_ip_int), port(_port) {}
};

class Strategy {
// public:
//     static std::shared_ptr<Strategy> getStrategy(std::string strategy_name) {
//         if (_strategy_maps.find(strategy_name) != _strategy_maps.end())
//             return _strategy_maps[strategy_name];
//         else
//             return nullptr;
//     }
// protected:
//     static std::unordered_map<std::string, std::shared_ptr<Strategy> > _strategy_maps;

public:
    Strategy() {};
    ~Strategy() {};
    virtual ServerSelection calculate(std::shared_ptr<MetadataManager> metadata, std::string model_name) = 0;
};

#endif