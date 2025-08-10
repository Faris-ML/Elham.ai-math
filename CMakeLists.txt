cmake_minimum_required(VERSION 3.10)
project(ElhamMath_cpp)

set(CMAKE_CXX_STANDARD 17)

# add_executable(ElhamMain test_code.cpp)
# target_link_libraries(ElhamMain PRIVATE ${PYTHON_LIBRARIES})

add_subdirectory(pybind11)  # ✅ this finds pybind11 locally
include_directories(${CMAKE_SOURCE_DIR})

pybind11_add_module(ElhamMath bindings.cpp)
