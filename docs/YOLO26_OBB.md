# YOLO26-OBB usage

**NOTE**: The yaml file is not required.

* [Convert model](#convert-model)
* [Compile the lib](#compile-the-lib)
* [Edit the config_infer_primary_yolo26_obb file](#edit-the-config_infer_primary_yolo26_obb-file)

##

### Convert model

#### 1. Download the Ultralytics repo and install the requirements

```
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics
pip3 install -e .
pip3 install onnx onnxslim onnxruntime
```

**NOTE**: It is recommended to use Python virtualenv.

#### 2. Copy conversor

Copy the `export_yolo26_obb.py` file from the `DeepStream-Yolo-OBB/utils` directory to the `ultralytics` folder.

#### 3. Download the model

Download the `pt` file from [YOLO26](https://github.com/ultralytics/assets/releases/) releases (example for YOLO26s-OBB)

```
wget https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-obb.pt
```

**NOTE**: You can use your custom YOLO26-OBB model.

#### 4. Convert model

Generate the ONNX model file (example for YOLO26s-OBB):

```
python3 export_yolo26_obb.py -w yolo26s-obb.pt --dynamic
```

**NOTE**: To change the inference size (default: 640)

```
-s SIZE
--size SIZE
-s HEIGHT WIDTH
--size HEIGHT WIDTH
```

Example for 1280:

```
-s 1280
```

or

```
-s 1280 1280
```

**NOTE**: To simplify the ONNX model

```
--simplify
```

**NOTE**: To use dynamic batch-size (DeepStream >= 6.1)

```
--dynamic
```

**NOTE**: To use static batch-size (example for batch-size = 4)

```
--batch 4
```

#### 5. Copy generated files

Copy the generated ONNX model file and `labels.txt` file (if generated) to the `DeepStream-Yolo-OBB` folder.

##

### Compile the lib

1. Open the `DeepStream-Yolo-OBB` folder and compile the lib.

2. Set the `CUDA_VER` according to your DeepStream version.

```
export CUDA_VER=XY.Z
```

* x86 platform

  ```
  DeepStream 9.0 = 13.1
  DeepStream 8.0 = 12.8
  DeepStream 7.1 = 12.6
  DeepStream 7.0 / 6.4 = 12.2
  ```

* Jetson platform

  ```
  DeepStream 8.0 = 13.0
  DeepStream 7.1 = 12.6
  DeepStream 7.0 / 6.4 = 12.2
  ```

3. Make the lib.

```
make -C nvdsinfer_custom_impl_Yolo_OBB clean && make -C nvdsinfer_custom_impl_Yolo_OBB
```

##

### Edit the config_infer_primary_yolo26_obb file

Edit the `config_infer_primary_yolo26_obb.txt` file according to your model (example for YOLO26s-OBB):

```
[property]
...
onnx-file=yolo26s-obb.onnx
...
num-detected-classes=80
...
network-type=3
...
parse-bbox-func-name=NvDsInferParseYoloOBB
...
```

If you use the CUDA parser entry point instead of the CPU parser, set:

```
parse-bbox-func-name=NvDsInferParseYoloOBBCuda
```

**NOTE**: The **DeepStream-Yolo-OBB** requires:

```
[property]
...
maintain-aspect-ratio=1
symmetric-padding=1
...
```
