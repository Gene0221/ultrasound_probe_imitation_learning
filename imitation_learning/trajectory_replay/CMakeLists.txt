cmake_minimum_required(VERSION 3.10)
project(ultrasound_franka_replay CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Franka REQUIRED)

add_executable(replay_trajectory_quintic src/replay_trajectory_quintic.cpp)
target_link_libraries(replay_trajectory_quintic Franka::Franka)

add_executable(replay_trajectory_bspline src/replay_trajectory_bspline.cpp)
target_link_libraries(replay_trajectory_bspline Franka::Franka)

if(MSVC)
  target_compile_options(replay_trajectory_quintic PRIVATE /W4)
  target_compile_options(replay_trajectory_bspline PRIVATE /W4)
else()
  target_compile_options(replay_trajectory_quintic PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(replay_trajectory_bspline PRIVATE -Wall -Wextra -Wpedantic)
endif()
