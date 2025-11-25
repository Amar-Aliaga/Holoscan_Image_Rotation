find_package(holoscan REQUIRED CONFIG)

add_library(pingOps_lib SHARED
    src/PingTxOp.cpp
    src/PingRxOp.cpp
    include/PingTxOp.hpp
    include/PingRxOp.hpp
)

target_include_directories(pingOps_lib PUBLIC include)

target_link_libraries(pingOps_lib PRIVATE holoscan::core)
