PROJECT (efa_adaptor)

aux_source_directory(. DIR_EFA_ADAPTOR_SRCS)
add_library( efa_adaptor ${DIR_EFA_ADAPTOR_SRCS})
target_link_libraries(efa_adaptor ${LIBFABRIC_LIBRARY} pthread rt)