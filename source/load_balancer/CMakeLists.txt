# CMakeLists for src directory
PROJECT(load_balancer)

AUX_SOURCE_DIRECTORY(. DIR_LOAD_BALANCER_SRCS)
ADD_EXECUTABLE ( load_balancer ${DIR_LOAD_BALANCER_SRCS})
TARGET_LINK_LIBRARIES( load_balancer utils;pthread )