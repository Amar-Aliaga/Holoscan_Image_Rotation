find_package(holoscan REQUIRED CONFIG)
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_CUDA_STANDARD 17) 
set(CMAKE_CUDA_ARCHITECTURES 70;75;80)  
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

enable_language(CUDA)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

add_library(pingMxOp_lib SHARED
    src/pingMxOp.cpp
    include/pingMxOp.hpp)

target_include_directories(pingMxOp_lib PRIVATE include
    /opt/nvidia/holoscan/include)

target_link_libraries(pingMxOp_lib PRIVATE
    holoscan::core
    holoscan::ops::holoviz)