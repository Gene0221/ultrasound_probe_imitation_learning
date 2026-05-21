#!/usr/bin/env python3
from __future__ import annotations

import json
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import Image


@dataclass
class RawPairSample:
    sample_id: int
    msg_a: Image
    msg_b: Image
    stamp_a: float
    stamp_b: float


@dataclass
class DecodedPairSample:
    sample_id: int
    image_a: np.ndarray
    image_b: np.ndarray
    stamp_a: float
    stamp_b: float


@dataclass
class SaveJob:
    index: int
    image_a: np.ndarray
    image_b: np.ndarray
    stamp_a: float
    stamp_b: float
    filename_a: str
    filename_b: str


class PairCaptureNode:
    def __init__(self) -> None:
        self.bridge = CvBridge()
        self.saved_count = 0
        self.received_count = 0
        self.last_previewed_sample_id = -1
        self.latest_raw_sample: Optional[RawPairSample] = None
        self.latest_decoded_sample: Optional[DecodedPairSample] = None
        self.latest_preview: Optional[np.ndarray] = None
        self.sample_lock = threading.Lock()

        self.topic_a = rospy.get_param("~camera_a/image_topic", rospy.get_param("/camera_a/image_topic"))
        self.topic_b = rospy.get_param("~camera_b/image_topic", rospy.get_param("/camera_b/image_topic"))
        self.queue_size = int(rospy.get_param("~synchronization/queue_size", rospy.get_param("/synchronization/queue_size", 5)))
        self.slop_seconds = float(rospy.get_param("~synchronization/slop_seconds", rospy.get_param("/synchronization/slop_seconds", 0.03)))
        self.subscriber_queue_size = int(rospy.get_param("~synchronization/subscriber_queue_size", 1))
        self.subscriber_buffer_bytes = int(
            rospy.get_param("~synchronization/subscriber_buffer_bytes", 2**24)
        )
        self.preview_fps = float(rospy.get_param("~capture/preview_fps", 10.0))
        self.preview_width = int(rospy.get_param("~capture/preview_width", rospy.get_param("/capture/preview_width", 1280)))
        self.image_extension = str(rospy.get_param("~capture/image_extension", rospy.get_param("/capture/image_extension", ".png")))
        self.jpeg_quality = int(rospy.get_param("~capture/jpeg_quality", 95))
        self.png_compression = int(rospy.get_param("~capture/png_compression", 3))
        self.session_name = str(rospy.get_param("~capture/session_name", rospy.get_param("/capture/session_name", "sample_session")))
        self.output_root = Path(
            str(rospy.get_param("~capture/output_root", rospy.get_param("/capture/output_root", "/tmp/two_camera_rgb_extrinsic")))
        )

        self.dataset_dir = self.output_root / self.session_name
        self.camera_a_dir = self.dataset_dir / "camera_a"
        self.camera_b_dir = self.dataset_dir / "camera_b"
        self.metadata_path = self.dataset_dir / "pairs_metadata.json"
        self.window_name = "Two-Camera RGB Capture"
        self.metadata: list[dict[str, float | int | str]] = []

        self.camera_a_dir.mkdir(parents=True, exist_ok=True)
        self.camera_b_dir.mkdir(parents=True, exist_ok=True)

        self.save_queue: queue.Queue[SaveJob | None] = queue.Queue(maxsize=32)
        self.save_worker = threading.Thread(target=self.save_worker_loop, daemon=True)
        self.save_worker.start()

        sub_a = Subscriber(
            self.topic_a,
            Image,
            queue_size=self.subscriber_queue_size,
            buff_size=self.subscriber_buffer_bytes,
        )
        sub_b = Subscriber(
            self.topic_b,
            Image,
            queue_size=self.subscriber_queue_size,
            buff_size=self.subscriber_buffer_bytes,
        )
        self.sync = ApproximateTimeSynchronizer([sub_a, sub_b], self.queue_size, self.slop_seconds)
        self.sync.registerCallback(self.callback)

    def callback(self, msg_a: Image, msg_b: Image) -> None:
        with self.sample_lock:
            self.received_count += 1
            self.latest_raw_sample = RawPairSample(
                sample_id=self.received_count,
                msg_a=msg_a,
                msg_b=msg_b,
                stamp_a=msg_a.header.stamp.to_sec(),
                stamp_b=msg_b.header.stamp.to_sec(),
            )

    def run(self) -> None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        rate = rospy.Rate(max(1.0, self.preview_fps))
        rospy.loginfo("Press S to save a synchronized pair, Q to quit.")

        try:
            while not rospy.is_shutdown():
                self.update_preview_if_needed()
                if self.latest_preview is not None:
                    cv2.imshow(self.window_name, self.latest_preview)
                key = cv2.waitKey(1) & 0xFF
                if key in {ord("q"), ord("Q")}:
                    break
                if key in {ord("s"), ord("S")}:
                    self.save_current_sample()
                rate.sleep()
        finally:
            self.flush_and_stop_saver()
            self.write_metadata()
            cv2.destroyAllWindows()

    def update_preview_if_needed(self) -> None:
        with self.sample_lock:
            raw_sample = self.latest_raw_sample
        if raw_sample is None or raw_sample.sample_id == self.last_previewed_sample_id:
            return

        image_a = self.bridge.imgmsg_to_cv2(raw_sample.msg_a, desired_encoding="bgr8")
        image_b = self.bridge.imgmsg_to_cv2(raw_sample.msg_b, desired_encoding="bgr8")
        decoded = DecodedPairSample(
            sample_id=raw_sample.sample_id,
            image_a=image_a,
            image_b=image_b,
            stamp_a=raw_sample.stamp_a,
            stamp_b=raw_sample.stamp_b,
        )
        self.latest_decoded_sample = decoded
        self.latest_preview = self.build_preview(decoded)
        self.last_previewed_sample_id = raw_sample.sample_id

    def build_preview(self, sample: DecodedPairSample) -> np.ndarray:
        panel_a = self.annotate(sample.image_a.copy(), "Camera A RGB")
        panel_b = self.annotate(sample.image_b.copy(), "Camera B RGB")
        width_each = max(1, self.preview_width // 2)
        panel_a = resize_to_width(panel_a, width_each)
        panel_b = resize_to_width(panel_b, width_each)
        target_height = min(panel_a.shape[0], panel_b.shape[0])
        panel_a = resize_to_height(panel_a, target_height)
        panel_b = resize_to_height(panel_b, target_height)
        preview = cv2.hconcat([panel_a, panel_b])
        status = (
            f"Received pairs: {self.received_count} | Saved pairs: {self.saved_count} | "
            "Keys: S save, Q quit"
        )
        cv2.putText(
            preview,
            status,
            (20, max(40, preview.shape[0] - 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return preview

    def annotate(self, image: np.ndarray, title: str) -> np.ndarray:
        cv2.putText(image, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        return image

    def save_current_sample(self) -> None:
        sample = self.latest_decoded_sample
        if sample is None:
            rospy.logwarn("No synchronized sample is available to save yet.")
            return

        next_index = self.saved_count + 1
        filename_a = f"camera_a_{next_index:04d}{self.image_extension}"
        filename_b = f"camera_b_{next_index:04d}{self.image_extension}"
        job = SaveJob(
            index=next_index,
            image_a=sample.image_a.copy(),
            image_b=sample.image_b.copy(),
            stamp_a=sample.stamp_a,
            stamp_b=sample.stamp_b,
            filename_a=filename_a,
            filename_b=filename_b,
        )
        try:
            self.save_queue.put_nowait(job)
        except queue.Full:
            rospy.logwarn("Save queue is full. Dropping this save request to keep capture responsive.")
            return

        self.saved_count = next_index
        time_delta = abs(sample.stamp_a - sample.stamp_b)
        self.metadata.append(
            {
                "pair_index": next_index,
                "camera_a_image": filename_a,
                "camera_b_image": filename_b,
                "camera_a_stamp": sample.stamp_a,
                "camera_b_stamp": sample.stamp_b,
                "timestamp_delta_seconds": time_delta,
            }
        )
        rospy.loginfo("Queued pair %04d for saving with timestamp delta %.6f s", next_index, time_delta)

    def save_worker_loop(self) -> None:
        while True:
            job = self.save_queue.get()
            if job is None:
                self.save_queue.task_done()
                break
            path_a = self.camera_a_dir / job.filename_a
            path_b = self.camera_b_dir / job.filename_b
            self.write_image(path_a, job.image_a)
            self.write_image(path_b, job.image_b)
            rospy.loginfo("Saved pair %04d to disk", job.index)
            self.save_queue.task_done()

    def write_image(self, path: Path, image: np.ndarray) -> None:
        suffix = path.suffix.lower()
        params: list[int] = []
        if suffix in {".jpg", ".jpeg"}:
            params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        elif suffix == ".png":
            params = [cv2.IMWRITE_PNG_COMPRESSION, self.png_compression]
        cv2.imwrite(str(path), image, params)

    def flush_and_stop_saver(self) -> None:
        self.save_queue.put(None)
        self.save_worker.join(timeout=10)

    def write_metadata(self) -> None:
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "session_name": self.session_name,
            "camera_a_topic": self.topic_a,
            "camera_b_topic": self.topic_b,
            "received_pair_count": self.received_count,
            "saved_pair_count": self.saved_count,
            "pairs": self.metadata,
        }
        self.metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        rospy.loginfo("Wrote metadata to %s", self.metadata_path)


def resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
    scale = width / image.shape[1]
    height = max(1, int(image.shape[0] * scale))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    scale = height / image.shape[0]
    width = max(1, int(image.shape[1] * scale))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def main() -> None:
    rospy.init_node("capture_rgb_pairs")
    node = PairCaptureNode()
    node.run()


if __name__ == "__main__":
    main()
