# CMakeLists for src directory
PROJECT(utils)

ADD_SUBDIRECTORY(tcp)
ADD_SUBDIRECTORY(tcp_pattern)
ADD_SUBDIRECTORY(shared_memory)
ADD_SUBDIRECTORY(memory_manager)
ADD_SUBDIRECTORY(efa_adaptor)

AUX_SOURCE_DIRECTORY(. DIR_UTILS_SRCS)
ADD_LIBRARY ( utils ${DIR_UTILS_SRCS})
TARGET_LINK_LIBRARIES( utils utils_tcp;utils_tcp_pattern;utils_shared_memory;utils_memory_manager;efa_adaptor )