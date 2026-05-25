#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import rospy
from geometry_msgs.msg import Quaternion, Vector3

from apros.msg import ProbePoseDelta


SCRIPT_PATH = Path(__file__).resolve()
TRACKING_ROOT = SCRIPT_PATH.parents[4]
DEFAULT_VISION_SCRIPT = TRACKING_ROOT / "scripts" / "vision" / "track_apriltag_pose_deltas.py"
DEFAULT_VISION_CONFIG = TRACKING_ROOT / "config" / "vision" / "apriltag_tracking.yaml"


def resolve_path(path_value: str, default_path: Path) -> Path:
    text = path_value.strip()
    if not text:
        return default_path.resolve()
    path = Path(text)
    if path.is_absolute():
        return path.resolve()
    return (TRACKING_ROOT / path).resolve()


def build_command(source: str) -> list[str]:
    python_executable = rospy.get_param("~python_executable", "").strip() or sys.executable
    if source == "vision":
        script_path = resolve_path(rospy.get_param("~vision_script", ""), DEFAULT_VISION_SCRIPT)
        config_path = resolve_path(rospy.get_param("~vision_config", ""), DEFAULT_VISION_CONFIG)
        return [
            python_executable,
            str(script_path),
            "--config",
            str(config_path),
            "--disable-jsonl",
            "--emit-stdout-records",
        ]

    if source == "fusion":
        fusion_command = rospy.get_param("~fusion_command", "").strip()
        if not fusion_command:
            raise ValueError("fusion source is selected, but ~fusion_command is empty.")
        return shlex.split(fusion_command)

    raise ValueError(f"Unsupported source: {source}")


def fill_message(source: str, frame_id: str, payload: dict[str, Any]) -> ProbePoseDelta:
    message = ProbePoseDelta()
    message.header.stamp = rospy.Time.from_sec(float(payload["prev_host_timestamp_s"]))
    message.header.frame_id = frame_id
    message.source = source
    message.tag_id = int(payload["tag_id"])
    message.valid = bool(payload.get("valid", True))
    message.prev_frame_number = int(payload["prev_frame_number"])
    message.curr_frame_number = int(payload["curr_frame_number"])
    message.prev_host_timestamp_s = float(payload["prev_host_timestamp_s"])
    message.curr_host_timestamp_s = float(payload["curr_host_timestamp_s"])
    message.prev_device_timestamp_ms = float(payload["prev_device_timestamp_ms"])
    message.curr_device_timestamp_ms = float(payload["curr_device_timestamp_ms"])

    translation = payload["delta_translation_xyz"]
    message.delta_translation = Vector3(
        x=float(translation[0]),
        y=float(translation[1]),
        z=float(translation[2]),
    )

    quaternion = payload["delta_quaternion_xyzw"]
    message.delta_quaternion = Quaternion(
        x=float(quaternion[0]),
        y=float(quaternion[1]),
        z=float(quaternion[2]),
        w=float(quaternion[3]),
    )

    transform = payload["delta_transform_prev_to_curr"]
    if len(transform) != 4 or any(len(row) != 4 for row in transform):
        raise ValueError("delta_transform_prev_to_curr must be a 4x4 matrix.")
    message.delta_transform_prev_to_curr = [float(value) for row in transform for value in row]
    return message


def forward_stderr(process: subprocess.Popen[str]) -> None:
    assert process.stderr is not None
    for raw_line in process.stderr:
        line = raw_line.rstrip()
        if line:
            rospy.loginfo("[provider] %s", line)


def run_bridge() -> None:
    source = rospy.get_param("~source", "vision").strip()
    topic_name = rospy.get_param("~topic_name", "/probe_pose_delta").strip()
    frame_id = rospy.get_param("~frame_id", "probe").strip()
    restart_on_exit = bool(rospy.get_param("~restart_on_exit", False))
    respawn_delay_s = float(rospy.get_param("~respawn_delay_s", 1.0))

    publisher = rospy.Publisher(topic_name, ProbePoseDelta, queue_size=100)

    while not rospy.is_shutdown():
        command = build_command(source)
        rospy.loginfo("Starting pose provider: %s", " ".join(command))

        process = subprocess.Popen(
            command,
            cwd=str(TRACKING_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

        stderr_thread = threading.Thread(target=forward_stderr, args=(process,), daemon=True)
        stderr_thread.start()

        try:
            assert process.stdout is not None
            for raw_line in process.stdout:
                if rospy.is_shutdown():
                    break
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                    message = fill_message(source, frame_id, payload)
                except Exception as exc:
                    rospy.logwarn("Skipping provider line: %s (%s)", line, exc)
                    continue
                publisher.publish(message)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()

        if rospy.is_shutdown():
            break

        return_code = process.returncode
        rospy.logwarn("Pose provider exited with code %s.", return_code)
        if not restart_on_exit:
            break

        rospy.sleep(max(0.0, respawn_delay_s))


def main() -> None:
    rospy.init_node("pose_delta_bridge")
    try:
        run_bridge()
    except Exception as exc:
        rospy.logerr("pose_delta_bridge failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
