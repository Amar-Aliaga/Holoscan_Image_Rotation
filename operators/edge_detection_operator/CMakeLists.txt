find_package(holoscan REQUIRED CONFIG)
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_CUDA_STANDARD 17) 
set(CMAKE_CUDA_ARCHITECTURES 70;75;80)  
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

enable_language(CUDA)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

add_library(edge_detection_lib SHARED
    src/edge_detection.cpp
    src/cuda_kernel.cu
    include/cuda_kernel.cuh
    include/edge_detection.hpp)

target_include_directories(edge_detection_lib PRIVATE include
    /opt/nvidia/holoscan/include)

target_link_libraries(edge_detection_lib PRIVATE
    holoscan::core
    holoscan::ops::holoviz
    CUDA::cudart
    ${OpenCV_LIBS}
    CUDA::nppc
    CUDA::nppif
    CUDA::nppig
    CUDA::nppicc
)
