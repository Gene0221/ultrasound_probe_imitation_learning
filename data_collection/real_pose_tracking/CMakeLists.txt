cmake_minimum_required(VERSION 3.16)
project(real_pose_tracking LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Franka REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Threads REQUIRED)

add_executable(read_franka_ee_pose
  src/read_franka_ee_pose.cpp
)

target_link_libraries(read_franka_ee_pose
  PRIVATE
    Franka::Franka
    yaml-cpp
    Threads::Threads
)
