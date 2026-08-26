#!/usr/bin/env python3
"""ROS 2 ONNX inference node that publishes candidate CTBR without PX4 authority."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from jetson_inference_trace import InferenceTraceWriter
from jetson_onnx_runtime import FastSlowOnnxRuntime


OBSERVATION_TOPIC = "/uav_payload/observation21"
CANDIDATE_TOPIC = "/uav_payload/rl_ctbr_candidate"
CONTEXT_TOPIC = "/uav_payload/rl_context"
STATUS_TOPIC = "/uav_payload/rl_inference_status"
DEFAULT_RUN_DIR_POINTER = "/tmp/vio_current_run_dir"
MANIFEST_RELATIVE_PATH = "config/manifest.json"


def resolve_run_dir(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    pointer = Path(DEFAULT_RUN_DIR_POINTER)
    if pointer.is_file():
        value = pointer.read_text(encoding="utf-8").strip()
        if value:
            return Path(value).expanduser().resolve()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return Path.home() / "uav_payload_runs" / stamp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--run-dir")
    parser.add_argument("--observation-topic", default=OBSERVATION_TOPIC)
    parser.add_argument("--candidate-topic", default=CANDIDATE_TOPIC)
    args, ros_args = parser.parse_known_args()

    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32MultiArray, String

    class FastSlowInferenceNode(Node):
        def __init__(self) -> None:
            super().__init__("rl_fastslow_inference_onnx_v1")
            self.runtime = FastSlowOnnxRuntime.from_bundle(args.bundle)
            run_dir = resolve_run_dir(args.run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            self.trace = InferenceTraceWriter(run_dir / "rl_inference_trace.csv")
            self.latest_observation: np.ndarray | None = None
            self.latest_rx_ns = 0
            self.latest_sequence = 0
            self.processed_sequence = 0
            self.stale_recorded = False
            self.candidate_pub = self.create_publisher(
                Float32MultiArray, args.candidate_topic, 10
            )
            self.context_pub = self.create_publisher(
                Float32MultiArray, CONTEXT_TOPIC, 10
            )
            self.status_pub = self.create_publisher(String, STATUS_TOPIC, 10)
            self.create_subscription(
                Float32MultiArray, args.observation_topic, self._on_observation, 20
            )
            self.timer = self.create_timer(1.0 / 60.0, self._on_timer)

        def _on_observation(self, message: Any) -> None:
            self.latest_observation = np.asarray(message.data, dtype=np.float32)
            self.latest_rx_ns = time.monotonic_ns()
            self.latest_sequence += 1
            self.stale_recorded = False

        def _stamp_and_write(self, record: dict[str, Any], rx_ns: int) -> None:
            now_wall = time.time_ns()
            now_mono = time.monotonic_ns()
            record.update(
                wall_time_ns=now_wall,
                monotonic_time_ns=now_mono,
                px4_timestamp_us=0,
                observation_timestamp_ns=rx_ns,
                input_sequence=self.processed_sequence,
            )
            self.trace.write(record)
            context = Float32MultiArray()
            context.data = [float(value) for value in record["context"]]
            self.context_pub.publish(context)
            status = String()
            status.data = json.dumps(
                {
                    "ready": bool(record["candidate_ready"]),
                    "valid": bool(record["observation_valid"]),
                    "history_fill": int(record["history_fill_count"]),
                    "valid_step": int(record["valid_step"]),
                    "reject_reason": record["reject_reason"],
                    "context_out_of_range": bool(record["context_out_of_range"]),
                    "context_severe_out_of_range": bool(
                        record["context_severe_out_of_range"]
                    ),
                    "latency_us": float(record["end_to_end_latency_us"]),
                },
                separators=(",", ":"),
            )
            self.status_pub.publish(status)

        def _on_timer(self) -> None:
            if self.latest_observation is None:
                return
            now_ns = time.monotonic_ns()
            age_sec = (now_ns - self.latest_rx_ns) * 1.0e-9
            if self.latest_sequence == self.processed_sequence:
                if age_sec > self.runtime.max_observation_age_sec and not self.stale_recorded:
                    record = self.runtime.reject(
                        "stale_observation", self.latest_observation, age_sec=age_sec
                    )
                    self._stamp_and_write(record, self.latest_rx_ns)
                    self.stale_recorded = True
                return
            self.processed_sequence = self.latest_sequence
            record = self.runtime.step(
                self.latest_observation, observation_age_sec=age_sec
            )
            self._stamp_and_write(record, self.latest_rx_ns)
            if record["candidate_ready"]:
                candidate = Float32MultiArray()
                candidate.data = [float(value) for value in record["action_clamped"]]
                self.candidate_pub.publish(candidate)

        def destroy_node(self) -> bool:
            self.trace.close()
            return super().destroy_node()

    rclpy.init(args=ros_args)
    node = FastSlowInferenceNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
