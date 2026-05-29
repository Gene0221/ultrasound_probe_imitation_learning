import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import serial
from serial import SerialException
from serial.tools import list_ports


PORT = "COM6"
BAUDRATE = 921600
RESOLUTION = 0.1  # raw LSB -> N
EXPECTED_SENSOR_COUNT = 2
SAMPLE_RATE_HZ = 30.0
ZERO_CHECK_SAMPLES = 30
CALIBRATION_FILE = Path(__file__).with_name("dp_s2015_calibration.json")
SENSOR_LABELS = [
    "DP-S2015 #1",
    "DP-S2015 #2",
]


def checksum(data: bytes) -> int:
    total = 0
    for byte in data:
        total = (total + byte) & 0xFF
    return ((total ^ 0xFF) + 1) & 0xFF


def le16(value: int) -> bytes:
    return int(value).to_bytes(2, "little", signed=False)


def build_hand_cmd(function_code: int, address: int, data_len: int, payload: bytes = b"") -> bytes:
    frame = bytearray([0x55, 0xAA, 0x00, function_code])
    frame += le16(address)
    frame += le16(data_len)
    frame += payload
    frame.append(checksum(frame))
    return bytes(frame)


def int8(value: int) -> int:
    return value - 256 if value > 127 else value


def parse_sensor_payload(payload: bytes, sensor_points: list[int]) -> list[dict]:
    sensors = []
    cursor = 0

    for sensor_index, point_count in enumerate(sensor_points):
        if cursor + 6 + point_count * 3 > len(payload):
            break

        combine_raw = payload[cursor:cursor + 6]
        cursor += 6

        fx = int.from_bytes(combine_raw[0:2], "little", signed=True)
        fy = int.from_bytes(combine_raw[2:4], "little", signed=True)
        fz = int.from_bytes(combine_raw[4:6], "little", signed=True)
        if fz < 0:
            fz = 256 + fz

        grid_raw = payload[cursor:cursor + point_count * 3]
        cursor += point_count * 3

        points = []
        for i in range(point_count):
            off = i * 3
            px = int8(grid_raw[off])
            py = int8(grid_raw[off + 1])
            pz = grid_raw[off + 2]
            points.append({
                "index": i + 1,
                "Fx": px * RESOLUTION,
                "Fy": py * RESOLUTION,
                "Fz": pz * RESOLUTION,
            })

        sensors.append({
            "sensor_index": sensor_index,
            "label": SENSOR_LABELS[sensor_index] if sensor_index < len(SENSOR_LABELS) else f"sensor{sensor_index}",
            "point_count": point_count,
            "total_force": {
                "Fx": fx * RESOLUTION,
                "Fy": fy * RESOLUTION,
                "Fz": fz * RESOLUTION,
            },
            "points": points,
        })

    return sensors


def apply_calibration(sensors: list[dict], calibration: dict | None) -> list[dict]:
    if not calibration:
        return sensors

    calibration_sensors = calibration.get("sensors", [])
    for sensor in sensors:
        idx = sensor["sensor_index"]
        if idx >= len(calibration_sensors):
            continue

        bias = calibration_sensors[idx]
        total_bias = bias.get("total_force", {})
        for axis in ("Fx", "Fy", "Fz"):
            sensor["total_force"][axis] -= float(total_bias.get(axis, 0.0))

        point_biases = bias.get("points", [])
        for point in sensor["points"]:
            point_idx = point["index"] - 1
            if point_idx >= len(point_biases):
                continue
            point_bias = point_biases[point_idx]
            for axis in ("Fx", "Fy", "Fz"):
                point[axis] -= float(point_bias.get(axis, 0.0))

    return sensors


def load_calibration(path=CALIBRATION_FILE) -> dict | None:
    if not path.exists():
        print(f"calibration file not found: {path}")
        return None
    with path.open("r", encoding="utf-8") as file:
        calibration = json.load(file)
    print(f"loaded calibration file: {path}")
    print(f"calibration created at: {calibration.get('created_at', 'unknown')}")
    return calibration


def save_calibration(calibration: dict, path=CALIBRATION_FILE):
    with path.open("w", encoding="utf-8") as file:
        json.dump(calibration, file, indent=2)
    print(f"saved calibration file: {path}")


def print_calibration_summary(calibration: dict):
    print("calibration result:")
    for sensor in calibration.get("sensors", []):
        total = sensor.get("total_force", {})
        print(
            f"  {sensor.get('label', 'sensor')}: "
            f"bias Fx={total.get('Fx', 0.0):+.2f}N, "
            f"bias Fy={total.get('Fy', 0.0):+.2f}N, "
            f"bias Fz={total.get('Fz', 0.0):+.2f}N, "
            f"max point Fz={sensor.get('max_point_fz', 0.0):.2f}N"
        )


class HandBoard:
    def __init__(self, port=PORT):
        self.port = port
        self.ser = None
        self.buffer = bytearray()
        self.sensor_points = []

    def connect(self):
        ports = [p.device for p in list_ports.comports()]
        print(f"available ports: {ports}")
        try:
            self.ser = serial.Serial(
                self.port,
                BAUDRATE,
                bytesize=8,
                stopbits=1,
                parity="N",
                timeout=0.05,
                write_timeout=0.2,
            )
        except SerialException as exc:
            raise RuntimeError(
                f"cannot open {self.port}. Available ports: {ports}. "
                "Close Paxini upper software or pass the correct port, for example: --port COM7"
            ) from exc
        time.sleep(0.3)
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        print(f"connected HAND board on {self.port}, baudrate={BAUDRATE}")

    def close(self):
        if self.ser and self.ser.is_open:
            try:
                self.send(build_hand_cmd(0x10, 23, 1, b"\x00"))
                time.sleep(0.1)
            finally:
                self.ser.close()
                print("disconnected")

    def send(self, frame: bytes):
        print(f"TX: {frame.hex(' ')}")
        self.ser.write(frame)
        self.ser.flush()

    def read_frame(self, timeout=1.0) -> bytes | None:
        deadline = time.perf_counter() + timeout

        while time.perf_counter() < deadline:
            chunk = self.ser.read(self.ser.in_waiting or 1)
            if chunk:
                self.buffer.extend(chunk)

            header_positions = []
            for header in (b"\xAA\x55", b"\xAA\x56"):
                idx = self.buffer.find(header)
                if idx >= 0:
                    header_positions.append(idx)

            if not header_positions:
                if len(self.buffer) > 1:
                    del self.buffer[:-1]
                continue

            start = min(header_positions)
            if start > 0:
                del self.buffer[:start]

            if len(self.buffer) < 8:
                continue

            if self.buffer[:2] == b"\xAA\x55":
                data_len = int.from_bytes(self.buffer[6:8], "little")
                total_len = 8 + data_len + 1
            elif self.buffer[:2] == b"\xAA\x56":
                stream_len = int.from_bytes(self.buffer[3:5], "little")
                total_len = stream_len + 6
            else:
                del self.buffer[0]
                continue

            if len(self.buffer) < total_len:
                continue

            frame = bytes(self.buffer[:total_len])
            del self.buffer[:total_len]

            if checksum(frame[:-1]) != frame[-1]:
                print(f"checksum mismatch, drop frame: {frame[:32].hex(' ')} ...")
                continue

            return frame

        return None

    def request(self, name: str, frame: bytes, timeout=1.0) -> bytes | None:
        self.send(frame)
        response = self.read_frame(timeout=timeout)
        if response is None:
            print(f"{name}: no response")
            return None
        print(f"RX {name}: {response.hex(' ')}")
        return response

    def get_version(self):
        response = self.request("version", build_hand_cmd(0x03, 0, 16))
        if not response:
            return None
        data_len = int.from_bytes(response[6:8], "little")
        version = response[8:8 + data_len].decode("utf-8", errors="ignore").strip("\x00")
        print(f"firmware version: {version}")
        return version

    def get_sensor_types(self):
        response = self.request("sensor_types", build_hand_cmd(0x03, 16, 4))
        if response:
            print(f"sensor types raw: {response[8:-1].hex(' ')}")

    def get_data_points(self):
        response = self.request("data_points", build_hand_cmd(0x03, 48, 68))
        if not response:
            return []

        data_len = int.from_bytes(response[6:8], "little")
        # The vendor app parses HAND normal responses from byte 7, not byte 8.
        # This keeps the 16-bit point-count words aligned as 00 34 -> 52.
        data = response[7:7 + data_len]
        points = []
        for i in range(0, len(data), 2):
            value = int.from_bytes(data[i:i + 2], "big")
            if value:
                points.append(value)

        if len(points) > EXPECTED_SENSOR_COUNT:
            print(f"raw non-empty sensor point counts: {points}")
            points = points[:EXPECTED_SENSOR_COUNT]

        self.sensor_points = points
        print(f"sensor point counts: {points}")
        detected = len(points)
        if detected < EXPECTED_SENSOR_COUNT:
            print(
                f"warning: expected {EXPECTED_SENSOR_COUNT} sensors, "
                f"but HAND board reported {detected}. "
                "Check the second DP-S2015 wiring/module setting."
            )
        else:
            labels = ", ".join(
                SENSOR_LABELS[i] if i < len(SENSOR_LABELS) else f"sensor{i}"
                for i in range(detected)
            )
            print(f"detected sensors: {labels}")
        return points

    def start_stream(self):
        self.request("auto_on", build_hand_cmd(0x10, 23, 1, b"\x01"), timeout=0.5)

    def stop_stream(self):
        self.request("auto_off", build_hand_cmd(0x10, 23, 1, b"\x00"), timeout=0.5)

    def calibrate_zero(self):
        print("calibrating zero, keep all DP-S2015 sensors unloaded...")
        response = self.request("calibration_zero", build_hand_cmd(0x17, 2, 1, b"\x01"), timeout=1.0)
        if response:
            time.sleep(0.5)
            print("calibration zero command sent")
        return response

    def read_stream_sensors(self, timeout=1.0):
        while True:
            frame = self.read_frame(timeout=timeout)
            if not frame:
                return None
            if frame[:2] == b"\xAA\x55":
                print(f"ACK: {frame.hex(' ')}")
                continue
            payload = frame[6:-1]
            return parse_sensor_payload(payload, self.sensor_points)

    def create_zero_calibration(self, samples=ZERO_CHECK_SAMPLES):
        if not self.sensor_points:
            self.get_data_points()
        if not self.sensor_points:
            raise RuntimeError("no sensor point metadata, cannot create calibration")

        print(f"collecting zero calibration with {samples} samples...")
        stats = [
            {
                "label": SENSOR_LABELS[i] if i < len(SENSOR_LABELS) else f"sensor{i}",
                "Fx": 0.0,
                "Fy": 0.0,
                "Fz": 0.0,
                "max_point_fz": 0.0,
                "count": 0,
                "points": [
                    {
                        "Fx": 0.0,
                        "Fy": 0.0,
                        "Fz": 0.0,
                        "count": 0,
                    }
                    for _ in range(point_count)
                ],
            }
            for i, point_count in enumerate(self.sensor_points)
        ]

        self.start_stream()
        try:
            for _ in range(samples):
                sensors = self.read_stream_sensors(timeout=1.0)
                if not sensors:
                    print("zero check timeout")
                    continue
                if len(sensors) < EXPECTED_SENSOR_COUNT:
                    print(
                        f"warning: expected {EXPECTED_SENSOR_COUNT} sensors in stream, "
                        f"but received {len(sensors)}"
                    )
                for sensor in sensors:
                    idx = sensor["sensor_index"]
                    if idx >= len(stats):
                        continue
                    total = sensor["total_force"]
                    max_point = max(sensor["points"], key=lambda p: p["Fz"]) if sensor["points"] else None
                    stats[idx]["Fx"] += total["Fx"]
                    stats[idx]["Fy"] += total["Fy"]
                    stats[idx]["Fz"] += total["Fz"]
                    stats[idx]["max_point_fz"] = max(
                        stats[idx]["max_point_fz"],
                        max_point["Fz"] if max_point else 0.0,
                    )
                    stats[idx]["count"] += 1
                    for point in sensor["points"]:
                        point_idx = point["index"] - 1
                        if point_idx >= len(stats[idx]["points"]):
                            continue
                        point_stats = stats[idx]["points"][point_idx]
                        point_stats["Fx"] += point["Fx"]
                        point_stats["Fy"] += point["Fy"]
                        point_stats["Fz"] += point["Fz"]
                        point_stats["count"] += 1
        finally:
            self.stop_stream()

        calibration = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "port": self.port,
            "baudrate": BAUDRATE,
            "sensor_points": self.sensor_points,
            "sample_count": samples,
            "sensors": [],
        }

        for item in stats:
            if item["count"] == 0:
                print(f"  {item['label']}: no valid samples")
                continue
            count = item["count"]
            sensor_bias = {
                "label": item["label"],
                "total_force": {
                    "Fx": item["Fx"] / count,
                    "Fy": item["Fy"] / count,
                    "Fz": item["Fz"] / count,
                },
                "max_point_fz": item["max_point_fz"],
                "points": [],
            }
            for point in item["points"]:
                point_count = point["count"] or 1
                sensor_bias["points"].append({
                    "Fx": point["Fx"] / point_count,
                    "Fy": point["Fy"] / point_count,
                    "Fz": point["Fz"] / point_count,
                })
            calibration["sensors"].append(sensor_bias)

        return calibration

    def check_zero_result(self, samples=ZERO_CHECK_SAMPLES):
        try:
            calibration = self.create_zero_calibration(samples=samples)
        except RuntimeError as exc:
            print(exc)
            return False
        print_calibration_summary(calibration)
        return len(calibration.get("sensors", [])) > 0

    def wait_for_start_confirmation(self):
        while True:
            choice = input("Start acquisition? Enter y to start, r to recalibrate, q to quit: ").strip().lower()
            if choice in ("y", "yes"):
                return "start"
            if choice in ("r", "retry"):
                return "retry"
            if choice in ("q", "quit"):
                return "quit"
            print("invalid input")

    def stream(self, duration=None, calibration=None):
        if not self.sensor_points:
            self.get_data_points()
        if not self.sensor_points:
            print("no sensor point metadata, cannot parse stream")
            return

        self.start_stream()
        start = time.perf_counter()
        next_sample_time = start
        sample_interval = 1.0 / SAMPLE_RATE_HZ
        count = 0

        try:
            while duration is None or time.perf_counter() - start < duration:
                sensors = self.read_stream_sensors(timeout=1.0)
                if not sensors:
                    print("stream timeout")
                    continue
                sensors = apply_calibration(sensors, calibration)
                now = time.perf_counter()
                if now < next_sample_time:
                    continue
                next_sample_time += sample_interval
                if next_sample_time < now - sample_interval:
                    next_sample_time = now + sample_interval
                count += 1

                parts = [f"[{count:04d}]"]
                for sensor in sensors:
                    total = sensor["total_force"]
                    max_point = max(sensor["points"], key=lambda p: p["Fz"]) if sensor["points"] else None
                    msg = (
                        f"{sensor['label']}({sensor['point_count']} points): "
                        f"Fx={total['Fx']:+.1f}N "
                        f"Fy={total['Fy']:+.1f}N "
                        f"Fz={total['Fz']:.1f}N"
                    )
                    if max_point:
                        msg += f" maxP{max_point['index']:02d}={max_point['Fz']:.1f}N"
                    parts.append(msg)
                print(" | ".join(parts))
        finally:
            self.stop_stream()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read DP-S2015 sensors through Paxini HAND board.")
    parser.add_argument("--port", default=PORT, help=f"serial port, default: {PORT}")
    args = parser.parse_args()

    board = HandBoard(port=args.port)
    try:
        board.connect()
        board.get_version()
        board.get_sensor_types()
        board.get_data_points()
        calibration = load_calibration()
        if calibration:
            print_calibration_summary(calibration)
        else:
            print("no calibration loaded; output will use raw sensor values")
        print(f"streaming continuously at {SAMPLE_RATE_HZ:.0f} Hz, press Ctrl+C to stop")
        board.stream(calibration=calibration)
    except KeyboardInterrupt:
        print("\nstop requested by user")
    finally:
        board.close()
