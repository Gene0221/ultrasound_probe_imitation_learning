#!/usr/bin/env python3
from __future__ import annotations

import os
import signal
import subprocess
import time

import rospy
from sensor_msgs.msg import Image


class DualD435iLauncher:
    def __init__(self) -> None:
        self.processes: list[subprocess.Popen] = []
        self.startup_timeout = float(rospy.get_param("~startup_timeout_seconds", 20.0))
        self.inter_camera_delay = float(rospy.get_param("~inter_camera_delay_seconds", 2.0))

        self.camera_a = {
            "serial_no": str(rospy.get_param("~serial_no_camera_a", "")),
            "camera_name": str(rospy.get_param("~camera_name_a", "camera_a")),
        }
        self.camera_b = {
            "serial_no": str(rospy.get_param("~serial_no_camera_b", "")),
            "camera_name": str(rospy.get_param("~camera_name_b", "camera_b")),
        }
        self.color_width = str(rospy.get_param("~color_width", 1920))
        self.color_height = str(rospy.get_param("~color_height", 1080))
        self.color_fps = str(rospy.get_param("~color_fps", 30))

    def run(self) -> None:
        try:
            self.launch_camera(self.camera_a)
            self.wait_for_topic(self.topic_for(self.camera_a["camera_name"]))
            rospy.loginfo(
                "Camera A is ready. Waiting %.1f seconds before starting camera B.",
                self.inter_camera_delay,
            )
            time.sleep(self.inter_camera_delay)

            self.launch_camera(self.camera_b)
            self.wait_for_topic(self.topic_for(self.camera_b["camera_name"]))
            rospy.loginfo("Camera B is ready.")

            rospy.loginfo("Dual D435i RGB sequential launcher is running.")
            rospy.spin()
        finally:
            self.shutdown_all()

    def launch_camera(self, camera_cfg: dict[str, str]) -> None:
        command = [
            "roslaunch",
            "realsense2_camera",
            "rs_camera.launch",
            f"camera:={camera_cfg['camera_name']}",
            f"serial_no:={camera_cfg['serial_no']}",
            "enable_color:=true",
            "enable_depth:=false",
            "enable_infra1:=false",
            "enable_infra2:=false",
            "enable_fisheye:=false",
            "enable_gyro:=false",
            "enable_accel:=false",
            f"color_width:={self.color_width}",
            f"color_height:={self.color_height}",
            f"color_fps:={self.color_fps}",
        ]
        if not camera_cfg["serial_no"]:
            rospy.logwarn(
                "No serial number provided for %s. Device mapping may become unstable across reboots or reconnects.",
                camera_cfg["camera_name"],
            )
        rospy.loginfo("Starting %s with command: %s", camera_cfg["camera_name"], " ".join(command))
        process = subprocess.Popen(command, preexec_fn=os.setsid)
        self.processes.append(process)

    def wait_for_topic(self, topic_name: str) -> None:
        rospy.loginfo("Waiting for topic: %s", topic_name)
        rospy.wait_for_message(topic_name, Image, timeout=self.startup_timeout)
        rospy.loginfo("Topic is active: %s", topic_name)

    def shutdown_all(self) -> None:
        for process in reversed(self.processes):
            if process.poll() is None:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGINT)
                except ProcessLookupError:
                    continue
        for process in reversed(self.processes):
            if process.poll() is None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    except ProcessLookupError:
                        continue

    @staticmethod
    def topic_for(camera_name: str) -> str:
        return f"/{camera_name}/color/image_raw"


def main() -> None:
    rospy.init_node("launch_dual_d435i_rgb")
    launcher = DualD435iLauncher()
    launcher.run()


if __name__ == "__main__":
    main()
