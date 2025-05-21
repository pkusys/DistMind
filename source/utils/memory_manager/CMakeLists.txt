AUX_SOURCE_DIRECTORY(. DIR_MEMORY_MANAGER_SRCS)
ADD_LIBRARY ( utils_memory_manager ${DIR_MEMORY_MANAGER_SRCS})
TARGET_LINK_LIBRARIES( utils_memory_manager utils_shared_memory )