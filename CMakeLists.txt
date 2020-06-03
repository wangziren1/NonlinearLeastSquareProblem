cmake_minimum_required(VERSION 2.8.3)
project(NonlinearLeastSquare)

set(CMAKE_BUILD_TYPE release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
find_package(yaml-cpp REQUIRED)
include_directories(${YAML_CPP_INCLUDE_DIR})

add_executable(NonlinearLeastSquare main.cpp)
target_link_libraries(NonlinearLeastSquare
  ${EIGEN3_LIBRARIES} ${YAML_CPP_LIBRARIES}
)