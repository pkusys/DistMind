# the whole CMakeLists.txt for project ttiger
PROJECT(pipeps)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

CMAKE_MINIMUM_REQUIRED(VERSION 3.2)
SET(CMAKE_CXX_STANDARD 17)
SET(CUDA_STANDARD 11)

if(UNIX AND NOT APPLE)
  # for Linux, BSD, Solaris, Minix
  # set(LIBFABRIC_LIBRARY /opt/amazon/efa/lib/libfabric.so)
  # set(LIBFABRIC_INCLUDE_DIR /opt/amazon/efa/include)
  set(LIBFABRIC_LIBRARY /opt/libfabric/lib/libfabric.so)
  set(LIBFABRIC_INCLUDE_DIR /opt/libfabric/include)
endif()

find_package(Python 3.10 COMPONENTS Interpreter Development REQUIRED)

set(CAFFE2_USE_CUDNN ON)
set(USE_CUSPARSELT OFF)

SET(CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/libtorch)
find_package(Torch REQUIRED)

# include directories
INCLUDE_DIRECTORIES(
  ${LIBFABRIC_INCLUDE_DIR}
  ${PROJECT_SOURCE_DIR}/source

  /usr/local/include
  /usr/include
)

# lib directories
LINK_DIRECTORIES(
  ${LIBFABRIC_LIBRARY}
  ${PROJECT_BINARY_DIR}/lib

  /usr/local/lib
  /use/lib
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -D_GLIBCXX_USE_CXX11_ABI=0")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options -fPIC")

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
SET(TEST_OUTPUT_PATH ${PROJECT_BINARY_DIR}/test)
SET(RESOURCE_PATH ${PROJECT_BINARY_DIR}/resource)

ADD_SUBDIRECTORY(pybind11)
ADD_SUBDIRECTORY(source)