# DeepStream-Yolo-OBB

NVIDIA DeepStream SDK 8.0 / 7.1 / 7.0 / 6.4 / 6.3 / 6.2 / 6.1.1 / 6.1 / 6.0.1 / 6.0 application for YOLO-OBB models

--------------------------------------------------------------------------------------------------
### This project is largely based on the following projects:
- https://github.com/marcoslucianops/DeepStream-Yolo
- https://github.com/marcoslucianops/DeepStream-Yolo-Pose
--------------------------------------------------------------------------------------------------

Since the DeepStream SDK does not natively support YOLO-OBB as a Detector (network-type=0) model, this project provides a workaround by treating YOLO-OBB as an Instance Segmentation model (network-type=3), enabling output of both the rotation angle and the four corners of each bounding box.

### Getting Started

* [Supported Models](#supported-models)
* [Basic Usage](#basic-usage)
* [YOLO11-OBB](docs/YOLO11_OBB.md)
* [YOLO26-OBB](docs/YOLO26_OBB.md)

### Supported Models

* [YOLO11-OBB](https://github.com/ultralytics/ultralytics)
* [YOLO26-OBB](https://github.com/ultralytics/ultralytics)

### Basic Usage

#### 1. Clone this repository
```bash
git clone https://github.com/samguns/DeepStream-Yolo-OBB.git
cd DeepStream-Yolo-OBB
```

#### 2. Compile the library and demo app

##### 2.1. Set the `CUDA_VER` according to your DeepStream version
```bash
export CUDA_VER=XY.Z
```

* X86 Platform

  ```
  DeepStream 9.0 = 13.1
  DeepStream 8.0 = 12.8
  DeepStream 7.1 = 12.6
  DeepStream 7.0 / 6.4 = 12.2
  ```

* Jetson Platform

  ```
  DeepStream 8.0 = 13.0
  DeepStream 7.1 = 12.6
  DeepStream 7.0 / 6.4 = 12.2
  ```

##### 2.2. Compile
```bash
make -C nvdsinfer_custom_impl_Yolo_OBB clean && make -C nvdsinfer_custom_impl_Yolo_OBB
make clean && make
```

#### 3. Run the application

# C code example

```bash
./deepstream -s file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_office.mp4 -c config_infer_primary_yolo26_obb.txt
```
