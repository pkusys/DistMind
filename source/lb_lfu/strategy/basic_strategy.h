#ifndef BALANCE_LB_BASIC_STRATEGY_H
#define BALANCE_LB_BASIC_STRATEGY_H

#include <unistd.h>

#include <vector>
#include <string>
#include <memory>
#include <iostream>

#include "strategy.h"

class BasicStrategy: public Strategy {
public:
    BasicStrategy(): Strategy() {/*_strategy_maps[std::string("basic")] = std::shared_ptr<BasicStrategy>(this);*/ std::cout << "Basic strategy" << std::endl;}
    ~BasicStrategy() {}
    
    ServerSelection calculate(std::shared_ptr<MetadataManager> metadata, std::string model_name);
};

#endif