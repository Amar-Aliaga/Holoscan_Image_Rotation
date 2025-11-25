find_package(holoscan REQUIRED CONFIG)

add_library(test_lib SHARED 
    include/test.hpp
    src/test.cpp)

target_include_directories(test_lib PRIVATE include
    /opt/nvidia/holoscan/include)

target_link_libraries(test_lib PRIVATE holoscan::core)