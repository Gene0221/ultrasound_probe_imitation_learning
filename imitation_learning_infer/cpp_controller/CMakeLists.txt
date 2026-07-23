cmake_minimum_required(VERSION 3.10)
project(ultrasound_realtime_controller CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Franka REQUIRED)
find_package(Threads REQUIRED)

add_executable(rolling_policy_controller src/rolling_policy_controller.cpp)
target_link_libraries(rolling_policy_controller Franka::Franka Threads::Threads)

if(MSVC)
  target_compile_options(rolling_policy_controller PRIVATE /W4)
else()
  target_compile_options(rolling_policy_controller PRIVATE -Wall -Wextra -Wpedantic)
endif()
