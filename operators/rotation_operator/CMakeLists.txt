find_package(holoscan REQUIRED CONFIG)
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_CUDA_STANDARD 17) 
set(CMAKE_CUDA_ARCHITECTURES 70;75;80)  
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

enable_language(CUDA)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

add_library(rotationOp_lib SHARED
    src/rotation_operator.cu
    src/cuda_rotate_image.cu
    include/rotation_operator.hpp
    include/cuda_rotate_image.cuh)

target_include_directories(rotationOp_lib PUBLIC include
    /opt/nvidia/holoscan/include
    /opt/nvidia/holoscan/include/gxf)

target_link_libraries(rotationOp_lib PRIVATE
    holoscan::core
    holoscan::ops::holoviz
    holoscan::ops::format_converter
    holoscan::ops::v4l2
    CUDA::cudart
    ${OpenCV_LIBS}
)

set_target_properties(rotationOp_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
