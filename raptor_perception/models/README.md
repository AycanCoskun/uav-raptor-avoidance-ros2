# Model

`bird_detector_node` loads `yolov5n.onnx` from this folder (installed to
`share/raptor_perception/models/`). The file is not tracked in git.

- Model: Ultralytics YOLOv5n (v7.0), COCO-pretrained, exported to ONNX.
  Input `images` [1, 3, 640, 640], output [1, 25200, 85]. Licence: AGPL-3.0.
- Target class: 14 (`bird`).

Copy an exported `yolov5n.onnx` here and rebuild the package, or pass another file with
`--ros-args -p model_path:=/path/to/model.onnx`.
