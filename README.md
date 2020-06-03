## Introduction
A Conversion tool to convert YOLO v3 Darknet weights to TF Lite model
(YOLO v3 in PyTorch > ONNX > TensorFlow > TF Lite).

## Prerequisites
- `python3`
- `torch==1.3.1`
- `torchvision==0.4.2`
- `onnx==1.6.0`
- `onnx-tf==1.5.0`
- `onnxruntime-gpu==1.0.0`
- `tensorflow-gpu==1.15.0`

## Docker
`docker pull zldrobit/onnx:10.0-cudnn7-devel`

## Usage
- **1. Download pretrained Darknet weights:**
```
cd weights
wget https://pjreddie.com/media/files/yolov3.weights 
```

- **2. Convert YOLO v3 model from Darknet weights to ONNX model:** 
Change `ONNX_EXPORT` to `True` in `models.py`. Run 
```
python3 detect.py --cfg cfg/yolov3.cfg --weights weights/yolov3.weights
```
The output ONNX file is `weights/export.onnx`.

- **3. Convert ONNX model to TensorFlow model:**
```
python3 onnx2tf.py
``` 
The output file is `weights/yolov3.pb`.

- **4. quantize_weights freeze_graph :**
```
python3 tf_opti.py
``` 
The output file is `weights/yolov3_opti.pb`.

- **5. add nms and convert to savedmodel**
```
python pb2savedmodel_batch.py
```
Now, you can run `python savedmodel_detect.py` to detect objects in an image.


## Acknowledgement
We borrow PyTorch code from [ultralytics/yolov3](https://github.com/ultralytics/yolov3), 
and TensorFlow low-level API conversion code from [paulbauriegel/tensorflow-tools](https://github.com/paulbauriegel/tensorflow-tools).
  
