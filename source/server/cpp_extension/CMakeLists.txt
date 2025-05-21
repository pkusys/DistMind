PROJECT(server_core)

AUX_SOURCE_DIRECTORY(. DIR_SERVER_CORE_SRCS)
ADD_LIBRARY ( server_core ${DIR_SERVER_CORE_SRCS})
TARGET_LINK_LIBRARIES( server_core utils)

add_subdirectory(torch)