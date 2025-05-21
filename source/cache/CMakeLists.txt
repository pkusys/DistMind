# CMakeLists for src directory
PROJECT(cache)

AUX_SOURCE_DIRECTORY(. DIR_CACHE_SRCS)
ADD_EXECUTABLE ( cache ${DIR_CACHE_SRCS})
TARGET_LINK_LIBRARIES( cache utils;pthread )