cmake_minimum_required(VERSION 3.0.2)
project(APROS)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  message_generation
  rospy
  std_msgs
)

add_message_files(
  FILES
  ProbePoseDelta.msg
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS
    geometry_msgs
    message_runtime
    rospy
    std_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/pose_delta_bridge.py
  DESTINATION ${catkin_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY
  config
  launch
  msg
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

