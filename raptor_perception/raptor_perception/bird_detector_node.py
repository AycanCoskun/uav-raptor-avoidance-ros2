# Copyright 2026 Aycan Coskun
# SPDX-License-Identifier: AGPL-3.0-only

"""
Bird detection node: YOLOv5n on the camera stream, one VisionDetection per frame.

Replaces the diag_live.csv output of the original yolo_trt_live_v18.py.
"""

import os
import time

from ament_index_python.packages import get_package_share_directory
import cv2
import numpy as np
from raptor_interfaces.msg import VisionDetection
from raptor_perception.yolo_detector import COCO_BIRD, NofMFilter, YoloV5Detector
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

TARGET_COLOUR = (0, 255, 0)
OTHER_COLOUR = (180, 180, 180)


def image_to_rgb(msg: Image):
    """Return the image as an RGB uint8 array, or None for unsupported encodings."""
    if msg.encoding not in ('rgb8', 'bgr8'):
        return None
    rows = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    image = rows[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
    if msg.encoding == 'bgr8':
        image = image[:, :, ::-1]
    return np.ascontiguousarray(image)


class BirdDetectorNode(Node):

    def __init__(self):
        super().__init__('bird_detector_node')
        model_path = self.declare_parameter('model_path', '').value
        if not model_path:
            model_path = os.path.join(
                get_package_share_directory('raptor_perception'), 'models', 'yolov5n.onnx')
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f'YOLO model not found: {model_path}')

        self._target_class = self.declare_parameter('target_class', COCO_BIRD).value
        conf_threshold = self.declare_parameter('conf_threshold', 0.35).value
        iou_threshold = self.declare_parameter('iou_threshold', 0.45).value
        confirm_n = self.declare_parameter('confirm_n', 2).value
        confirm_m = self.declare_parameter('confirm_m', 3).value
        num_threads = self.declare_parameter('num_threads', 2).value
        self._publish_debug = self.declare_parameter('publish_debug_image', True).value

        self._detector = YoloV5Detector(model_path, conf_threshold, iou_threshold, num_threads)
        self._filter = NofMFilter(confirm_n, confirm_m)
        self._warned_encoding = False

        # Depth 1: always process the newest frame and drop stale ones
        # (same idea as BUFFERSIZE=1 and the size-limited frame queue in v18).
        camera_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._detection_pub = self.create_publisher(VisionDetection, 'perception/detection', 10)
        self._debug_pub = self.create_publisher(Image, 'perception/debug_image', 1)
        self._image_sub = self.create_subscription(
            Image, 'camera/image_raw', self._on_image, camera_qos)

        self.get_logger().info(
            f'model={model_path} target_class={self._target_class} '
            f'conf_threshold={conf_threshold:.2f} N={confirm_n} M={confirm_m} '
            f'threads={num_threads}')

    def _on_image(self, msg: Image) -> None:
        rgb = image_to_rgb(msg)
        if rgb is None:
            if not self._warned_encoding:
                self.get_logger().warning(f'Unsupported image encoding: {msg.encoding}')
                self._warned_encoding = True
            return

        start = time.perf_counter()
        detections = self._detector.detect(rgb)
        inference_ms = (time.perf_counter() - start) * 1000.0
        targets = [d for d in detections if d.cls == self._target_class]

        out = VisionDetection()
        out.header = msg.header
        out.hazard = bool(targets)
        out.confirmed = self._filter.update(out.hazard)
        out.max_conf = 0.0
        out.cx_norm = 0.5  # image centre when nothing is detected, as in state_machine_v3
        out.cy_norm = 0.5
        if targets:
            best = max(targets, key=lambda d: d.conf)
            out.max_conf = float(best.conf)
            out.cx_norm = float((best.x1 + best.x2) / 2.0 / msg.width)
            out.cy_norm = float((best.y1 + best.y2) / 2.0 / msg.height)
        out.inference_ms = float(inference_ms)
        self._detection_pub.publish(out)

        if self._publish_debug and self._debug_pub.get_subscription_count() > 0:
            self._debug_pub.publish(self._debug_image(msg, rgb, detections, out))

        self.get_logger().info(
            f'max_conf={out.max_conf:.2f} hazard={int(out.hazard)} '
            f'confirmed={int(out.confirmed)} cx={out.cx_norm:.2f} cy={out.cy_norm:.2f} '
            f'inference={inference_ms:.1f} ms',
            throttle_duration_sec=2.0)

    def _debug_image(self, msg, rgb, detections, out) -> Image:
        canvas = rgb.copy()
        for d in detections:
            colour = TARGET_COLOUR if d.cls == self._target_class else OTHER_COLOUR
            cv2.rectangle(canvas, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), colour, 2)
            cv2.putText(canvas, f'{d.cls}:{d.conf:.2f}', (int(d.x1), max(12, int(d.y1) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
        status = 'CONFIRMED' if out.confirmed else ('DETECTED' if out.hazard else 'CLEAR')
        cv2.putText(canvas, f'{status}  {out.inference_ms:.0f} ms', (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        image = Image()
        image.header = msg.header
        image.height, image.width = canvas.shape[:2]
        image.encoding = 'rgb8'
        image.step = image.width * 3
        image.data.frombytes(canvas.tobytes())
        return image


def main(args=None):
    rclpy.init(args=args)
    try:
        node = BirdDetectorNode()
    except (FileNotFoundError, ValueError) as error:
        rclpy.logging.get_logger('bird_detector_node').fatal(str(error))
        rclpy.try_shutdown()
        return 1
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        # A callback that is still running when SIGINT/SIGTERM shuts the context
        # down fails to publish; that is a normal shutdown, anything else is not.
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == '__main__':
    main()
