cmake_minimum_required(VERSION 3.0.2)
project(two_camera_rgb_extrinsic)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  message_filters
  roscpp
  rospy
  sensor_msgs
  std_msgs
)

find_package(OpenCV REQUIRED)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(capture_rgb_pairs_node
  src/capture_rgb_pairs_node.cpp
)

target_link_libraries(capture_rgb_pairs_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

catkin_install_python(PROGRAMS
  scripts/capture_rgb_pairs.py
  scripts/estimate_rgb_extrinsics.py
  scripts/launch_dual_d435i_rgb.py
  scripts/print_realsense_devices.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(TARGETS capture_rgb_pairs_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
