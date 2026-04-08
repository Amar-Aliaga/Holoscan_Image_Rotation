find_package(holoscan REQUIRED CONFIG)
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(holoscan_app main.cpp)

target_include_directories(holoscan_app PRIVATE
    ${CMAKE_SOURCE_DIR}/operators/rotation_operator/include
    ${CMAKE_SOURCE_DIR}/operators/ping_multiplier_operator/include
    ${CMAKE_SOURCE_DIR}/operators/ping_ops/include
    ${CMAKE_SOURCE_DIR}/operators/test_operator/include
    ${CMAKE_SOURCE_DIR}/operators/edge_detection_operator/include
    ${CMAKE_SOURCE_DIR}/operators/blur_operator/include
    /opt/nvidia/holoscan/include
    /workspace/expected/include
)

target_link_libraries(holoscan_app PRIVATE
    holoscan::core
    holoscan::ops::holoviz
    holoscan::ops::format_converter
    holoscan::ops::v4l2             
    CUDA::cudart
    ${OpenCV_LIBS}
    rotationOp_lib
    pingMxOp_lib
    pingOps_lib
    test_lib
    edge_detection_lib
    blur_lib
)

set_target_properties(holoscan_app PROPERTIES CUDA_SEPARABLE_COMPILATION ON)