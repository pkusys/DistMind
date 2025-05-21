# CMakeLists for src directory
PROJECT(lb_lfu)

ADD_SUBDIRECTORY(strategy)

AUX_SOURCE_DIRECTORY(. DIR_LOAD_BALANCER_SRCS)
ADD_EXECUTABLE ( lb_lfu ${DIR_LOAD_BALANCER_SRCS})
TARGET_LINK_LIBRARIES( lb_lfu lb_strategy;utils;pthread )