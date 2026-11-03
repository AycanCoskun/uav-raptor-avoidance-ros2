# Copyright 2026 Aycan Coskun
# SPDX-License-Identifier: AGPL-3.0-only

import numpy as np
import pytest

from raptor_perception.yolo_detector import (
    COCO_BIRD, decode, Detection, iou, letterbox, nms, NofMFilter)


def make_row(cx, cy, w, h, obj, cls, score):
    row = np.zeros(85, dtype=np.float32)
    row[:5] = [cx, cy, w, h, obj]
    row[5 + cls] = score
    return row


def test_letterbox_keeps_aspect_ratio_and_pads():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    padded, scale, pad_x, pad_y = letterbox(image, 640)
    assert padded.shape == (640, 640, 3)
    assert scale == pytest.approx(1.0)
    assert (pad_x, pad_y) == (0, 80)
    assert padded[0, 0, 0] == 114  # padding colour
    assert padded[80, 0, 0] == 0   # image area


def test_letterbox_scales_down_wide_images():
    image = np.zeros((720, 1280, 3), dtype=np.uint8)  # ZED left eye, as in v18
    _, scale, pad_x, pad_y = letterbox(image, 640)
    assert scale == pytest.approx(0.5)
    assert (pad_x, pad_y) == (0, 140)


def test_decode_uses_objectness_times_class_score():
    pred = np.stack([
        make_row(320, 320, 40, 20, 0.9, COCO_BIRD, 0.8),  # conf 0.72, kept
        make_row(100, 100, 10, 10, 0.9, COCO_BIRD, 0.3),  # conf 0.27, dropped
        make_row(200, 200, 10, 10, 0.2, 0, 1.0),          # objectness too low
    ])
    dets = decode(pred, 0.35, 1.0, 0, 0, 640, 640)
    assert len(dets) == 1
    assert dets[0].cls == COCO_BIRD
    assert dets[0].conf == pytest.approx(0.72)
    assert (dets[0].x1, dets[0].y1, dets[0].x2, dets[0].y2) == pytest.approx((300, 310, 340, 330))


def test_decode_undoes_letterbox_and_clips():
    # 640x480 image letterboxed into 640x640: pad_y = 80, scale = 1.
    pred = np.stack([make_row(10, 90, 40, 40, 1.0, COCO_BIRD, 1.0)])
    dets = decode(pred, 0.35, 1.0, 0, 80, 640, 480)
    assert (dets[0].x1, dets[0].y1, dets[0].x2, dets[0].y2) == pytest.approx((0, 0, 30, 30))


def test_nms_suppresses_overlaps_of_the_same_class_only():
    dets = [
        Detection(0, 0, 10, 10, 0.9, COCO_BIRD),
        Detection(1, 1, 11, 11, 0.8, COCO_BIRD),  # overlaps the first bird
        Detection(1, 1, 11, 11, 0.7, 0),          # same box, other class
        Detection(50, 50, 60, 60, 0.6, COCO_BIRD),
    ]
    kept = nms(dets, 0.45)
    assert [(d.conf, d.cls) for d in kept] == [(0.9, COCO_BIRD), (0.7, 0), (0.6, COCO_BIRD)]


def test_iou():
    a = Detection(0, 0, 10, 10, 1.0, 0)
    assert iou(a, a) == pytest.approx(1.0)
    assert iou(a, Detection(5, 0, 15, 10, 1.0, 0)) == pytest.approx(50 / 150)
    assert iou(a, Detection(20, 20, 30, 30, 1.0, 0)) == 0.0


def test_n_of_m_two_of_three():
    nofm = NofMFilter(2, 3)
    assert not nofm.update(True)
    assert not nofm.update(True)   # window not full yet
    assert nofm.update(False)      # [1 1 0]
    assert not nofm.update(False)  # [1 0 0]
    assert not nofm.update(True)   # [0 0 1] single-frame detection
    assert nofm.update(True)       # [0 1 1]


def test_n_of_m_rejects_invalid_config():
    with pytest.raises(ValueError):
        NofMFilter(0, 3)
    with pytest.raises(ValueError):
        NofMFilter(4, 3)
