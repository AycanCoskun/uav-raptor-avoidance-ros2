# Model

`bird_detector_node` loads `yolov5n.onnx` from this folder (installed to
`share/raptor_perception/models/`). The file is not tracked in git.

- Model: Ultralytics YOLOv5n (v7.0), COCO-pretrained, exported to ONNX.
  Input `images` [1, 3, 640, 640], output [1, 25200, 85] (no NMS in the graph). Licence: AGPL-3.0.
- Target class: 14 (`bird`).

The results in this repository were obtained with the ONNX file the original project's TensorRT
engines were built from. The standard Ultralytics export produces a model in the same format
(not re-verified for this repository):

```bash
git clone https://github.com/ultralytics/yolov5.git && cd yolov5 && git checkout v7.0
pip install -r requirements.txt
python export.py --weights yolov5n.pt --include onnx --imgsz 640
```

Copy `yolov5n.onnx` here and rebuild the package, or pass another file with
`--ros-args -p model_path:=/path/to/model.onnx`.
