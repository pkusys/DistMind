# CMakeLists for src directory
PROJECT(metadata_storage)

AUX_SOURCE_DIRECTORY(. DIR_METADATA_STORAGE_SRCS)
ADD_EXECUTABLE ( metadata_storage ${DIR_METADATA_STORAGE_SRCS})
TARGET_LINK_LIBRARIES( metadata_storage utils )