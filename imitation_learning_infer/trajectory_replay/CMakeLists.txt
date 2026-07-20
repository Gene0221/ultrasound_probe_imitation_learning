cmake_minimum_required(VERSION 3.10)
project(ultrasound_franka_replay CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Franka REQUIRED)

add_executable(replay_trajectory src/replay_trajectory.cpp)
target_link_libraries(replay_trajectory Franka::Franka)

if(MSVC)
  target_compile_options(replay_trajectory PRIVATE /W4)
else()
  target_compile_options(replay_trajectory PRIVATE -Wall -Wextra -Wpedantic)
endif()
