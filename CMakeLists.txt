cmake_minimum_required(VERSION 3.20)
project(holoscan_app LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(operators/rotation_operator)
add_subdirectory(operators/ping_multiplier_operator)
add_subdirectory(operators/ping_ops)
add_subdirectory(operators/test_operator)
add_subdirectory(app)


# sudo docker run --rm -it --gpus all \
#   --device /dev/video0 \
#   -e DISPLAY=$DISPLAY \
#   -v /tmp/.X11-unix:/tmp/.X11-unix \
#   -v /opt/nvidia/holoscan:/opt/nvidia/holoscan \
#   -v /home/amar-aliaga/holoscan/first_holoscan_program:/workspace \
#   my-holoscan-app