PROJECT(storage_client)

if(NOT TARGET spdlog)
    # Stand-alone build
    find_package(spdlog REQUIRED)
endif()

# AUX_SOURCE_DIRECTORY(. DIR_STORAGE_CLIENT_SRCS)
ADD_EXECUTABLE ( storage_client store_cli.cpp)
ADD_EXECUTABLE ( storage_stress_test stress_test.cpp cli_common.hpp)
ADD_EXECUTABLE ( shard_deploy deploy_bins.cpp cli_common.hpp)
ADD_EXECUTABLE ( shard_cli shard_client.cpp cli_common.hpp)
ADD_EXECUTABLE ( shard_ctl shard_ctl.cpp cli_common.hpp)
TARGET_LINK_LIBRARIES( storage_client utils )
TARGET_LINK_LIBRARIES(storage_stress_test PRIVATE spdlog utils)
TARGET_LINK_LIBRARIES(shard_deploy PRIVATE spdlog utils)
TARGET_LINK_LIBRARIES(shard_cli PRIVATE spdlog utils)
TARGET_LINK_LIBRARIES(shard_ctl PRIVATE spdlog utils)