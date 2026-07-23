from __future__ import annotations

import argparse
import json
import socket
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mock TCP controller for testing policy sender JSONL chunks.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50555)
    return parser.parse_args()


def validate_payload(payload: dict[str, Any]) -> None:
    actions = payload.get("actions")
    if not isinstance(actions, list) or len(actions) == 0:
        raise ValueError("actions must be a non-empty list")
    for index, action in enumerate(actions):
        if not isinstance(action, list) or len(action) != 7:
            raise ValueError(f"actions[{index}] must have 7 values")
        for value in action:
            float(value)
    if "force_safety_ok" not in payload:
        raise ValueError("force_safety_ok is missing")


def main() -> None:
    args = parse_args()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(1)
        print(f"[INFO] Mock controller listening on {args.host}:{args.port}")
        conn, addr = server.accept()
        print(f"[INFO] Client connected: {addr}")
        with conn:
            buffer = ""
            count = 0
            while True:
                data = conn.recv(4096)
                if not data:
                    break
                buffer += data.decode("utf-8")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if not line:
                        continue
                    payload = json.loads(line)
                    validate_payload(payload)
                    count += 1
                    print(
                        f"[INFO] chunk={count} seq={payload.get('seq')} "
                        f"actions={len(payload['actions'])} force_ok={payload.get('force_safety_ok')}"
                    )


if __name__ == "__main__":
    main()
