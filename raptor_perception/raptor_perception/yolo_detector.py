# Copyright 2026 Aycan Coskun
# SPDX-License-Identifier: AGPL-3.0-only

"""
YOLOv5n inference with ONNX Runtime.

ROS-independent port of the TensorRT pipeline in yolo_trt_live_v18.py: letterbox to
640x640 (pad value 114), RGB / 255 input, decode of the (25200, 85) output with
conf = objectness * class score, and greedy per-class NMS.
"""

from collections import deque
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime as ort

COCO_BIRD = 14


@dataclass
class Detection:
    """One bounding box in original image pixels."""

    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls: int


def letterbox(
        image: np.ndarray, size: int = 640, pad_value: int = 114
) -> Tuple[np.ndarray, float, int, int]:
    """
    Resize keeping the aspect ratio and pad to size x size.

    Returns the padded image, the scale factor and the x / y padding in pixels.
    """
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_w = int(round(width * scale))
    new_h = int(round(height * scale))
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.full((size, size, 3), pad_value, dtype=np.uint8)
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return padded, scale, pad_x, pad_y


def decode(
        pred: np.ndarray, conf_threshold: float, scale: float, pad_x: float, pad_y: float,
        width: int, height: int, max_det: int = 400
) -> List[Detection]:
    """Convert raw YOLOv5 rows [cx, cy, w, h, obj, 80 class scores] to detections."""
    pred = pred[pred[:, 4] >= conf_threshold]
    if pred.shape[0] == 0:
        return []
    class_scores = pred[:, 5:]
    cls = np.argmax(class_scores, axis=1)
    conf = pred[:, 4] * class_scores[np.arange(pred.shape[0]), cls]
    keep = conf >= conf_threshold
    pred, cls, conf = pred[keep], cls[keep], conf[keep]

    half_w = pred[:, 2] / 2.0
    half_h = pred[:, 3] / 2.0
    x1 = np.clip((pred[:, 0] - half_w - pad_x) / scale, 0.0, width)
    y1 = np.clip((pred[:, 1] - half_h - pad_y) / scale, 0.0, height)
    x2 = np.clip((pred[:, 0] + half_w - pad_x) / scale, 0.0, width)
    y2 = np.clip((pred[:, 1] + half_h - pad_y) / scale, 0.0, height)
    valid = (x2 > x1) & (y2 > y1)

    order = np.argsort(-conf[valid])[:max_det]
    boxes = np.stack([x1, y1, x2, y2], axis=1)[valid][order]
    return [
        Detection(float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(c), int(k))
        for b, c, k in zip(boxes, conf[valid][order], cls[valid][order])
    ]


def iou(a: Detection, b: Detection) -> float:
    """Intersection over union of two boxes."""
    inter_w = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    inter_h = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
    return inter / union if union > 0.0 else 0.0


def nms(detections: List[Detection], iou_threshold: float = 0.45) -> List[Detection]:
    """Greedy per-class non-maximum suppression."""
    remaining = sorted(detections, key=lambda d: d.conf, reverse=True)
    kept = []
    while remaining:
        best = remaining.pop(0)
        kept.append(best)
        remaining = [
            d for d in remaining if d.cls != best.cls or iou(best, d) < iou_threshold
        ]
    return kept


class NofMFilter:
    """Confirmed once the window holds M flags and at least N of them are set."""

    def __init__(self, n: int, m: int):
        if n < 1 or n > m:
            raise ValueError('N-of-M filter requires 1 <= N <= M')
        self._n = n
        self._m = m
        self._window = deque(maxlen=m)

    def update(self, flag: bool) -> bool:
        self._window.append(bool(flag))
        return len(self._window) == self._m and sum(self._window) >= self._n

    def reset(self) -> None:
        self._window.clear()


class YoloV5Detector:
    """YOLOv5 ONNX model running on the ONNX Runtime CPU execution provider."""

    def __init__(
            self, model_path: str, conf_threshold: float = 0.35, iou_threshold: float = 0.45,
            num_threads: int = 2):
        options = ort.SessionOptions()
        options.intra_op_num_threads = num_threads
        self._session = ort.InferenceSession(
            model_path, sess_options=options, providers=['CPUExecutionProvider'])
        model_input = self._session.get_inputs()[0]
        self._input_name = model_input.name
        size = model_input.shape[2]
        self.input_size = size if isinstance(size, int) else 640
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def detect(self, rgb: np.ndarray) -> List[Detection]:
        """Run the model on an RGB uint8 image and return detections in image pixels."""
        height, width = rgb.shape[:2]
        padded, scale, pad_x, pad_y = letterbox(rgb, self.input_size)
        blob = padded.transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255.0
        pred = self._session.run(None, {self._input_name: blob})[0][0]
        detections = decode(pred, self.conf_threshold, scale, pad_x, pad_y, width, height)
        return nms(detections, self.iou_threshold)
