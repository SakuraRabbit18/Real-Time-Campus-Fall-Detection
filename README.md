# Real-Time-Campus-Fall-Detection
Real time fall detection project for college students in physical education classes

## 1. Project Overview

This project is built on the Ultralytics YOLO series algorithms and focuses on human fall detection in campus physical education class scenarios. By constructing a dedicated campus PE class fall detection dataset (CampusFallDetectionDataset) and leveraging the efficient object detection capabilities of YOLO models, it enables real-time recognition and early warning of falls during physical education activities. It can be applied to safety monitoring in PE class venues such as playgrounds, gymnasiums, and sports fields, providing technical support for student safety during physical education.

## 2. Project Structure

The project is divided into two main parts: **dataset directory** and **algorithm framework directory**, with a clear and concise structure for easy location of core files:

```plaintext
C:.
├─CampusFallDetectionDataset  # Dedicated dataset for campus fall detection (core data storage)
│  ├─images                   # Image file directory (divided by training/validation/testing)
│  │  ├─test                  # Test set images (for final model performance evaluation)
│  │  ├─train                 # Training set images (for model parameter learning)
│  │  └─val                   # Validation set images (for performance verification during training)
│  └─labels                   # Label file directory (corresponding to image directories, YOLO format)
│      ├─test                  # Test set labels (.txt format, matching corresponding images)
│      ├─train                 # Training set labels
│      └─val                   # Validation set labels
└─ultralytics                 # Ultralytics YOLO algorithm framework (core code and tools)
    ├─datasets                 # Framework default dataset configurations (referencable for customization)
    ├─docs                     # Official documentation (tutorials, API references, deployment guides)
    ├─examples                 # Model deployment examples (multi-language/multi-framework, e.g., C++, ONNX Runtime)
    ├─runs                     # Model run records (training logs, weight files, visualization results)
    ├─tests                    # Framework test code (unit tests, integration tests)
    └─ultralytics              # Framework core code (model definitions, training engine, utility functions, etc.)
```

## 3. Core Technologies

The core technology stack of this project is built around **object detection** and **scenario-specific datasets**, including:

| Technology Category        | Details                                                      |
| -------------------------- | ------------------------------------------------------------ |
| Object Detection Framework | Ultralytics YOLO (supports YOLOv8/YOLO12/RT-DETR and other models, efficient, user-friendly, and extensible) |
| Dataset Format             | YOLO label format (.txt files, each line contains "class ID + object center coordinates + width and height") |
| Model Deployment Tools     | ONNX Runtime, OpenVINO, TFLite, CPP/Rust cross-language deployment |

## 4. Recommended Environment

To ensure stable operation of the project, the following hardware and software environment configurations are recommended:

| Software/Library | Recommended Version          | Description                                      |
| ---------------- | ---------------------------- | ------------------------------------------------ |
| Operating System | Windows 10/11, Ubuntu 20.04+ | Compatible with mainstream development systems   |
| Python           | 3.8 - 3.11                   | Supported Python version range for the framework |
| PyTorch          | 2.0+                         | Core framework for model training and inference  |
| Ultralytics      | 8.0+                         | Official implementation library for YOLO models  |
| OpenCV-Python    | 4.5+                         | Image reading, preprocessing, and visualization  |
| NumPy            | 1.21+                        | Numerical computation and array processing       |
| ONNX Runtime     | 1.15+                        | Inference acceleration for ONNX models           |

**Environment Installation Steps**

1. **Create a virtual environment** (Conda or venv recommended):

   ```bash
   # Conda example
   conda create -n campus_fall python=3.10
   conda activate campus_fall
   ```

2. **Install PyTorch** (choose based on CUDA version; install CPU version if no GPU available):

   ```bash
   # Example for CUDA 11.8 (NVIDIA driver required)
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   # CPU version
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

3. **Install project dependencies**:

   ```bash
   # Install Ultralytics framework
   pip install ultralytics
   # Install other dependencies
   pip install opencv-python numpy onnxruntime
   ```

## 5. Dataset

### 5.1 Dataset Introduction

CampusFallDetectionDataset is a scenario-specific dataset designed for fall detection in campus physical education classes,

### 5.2 Label Format Explanation

YOLO label files (e.g., `image_001.txt`) have the following format (coordinates are normalized relative to image width and height):

txt

```txt
0 0.52 0.38 0.12 0.24  # Class 0 (normal behavior): center coordinates (0.52, 0.38), width×height 0.12×0.24
1 0.78 0.65 0.15 0.30  # Class 1 (fall behavior): center coordinates (0.78, 0.65), width×height 0.15×0.30
```

### 5.3 Dataset Configuration

To train a model using this dataset, create a custom configuration file (e.g., `campus_fall.yaml`) in the `ultralytics/cfg/datasets/` directory with the following content:

```yaml
# Dataset paths (modify according to actual paths)
path: ../CampusFallDetectionDataset  # Dataset root directory
train: images/train                  # Training set images path (relative to path)
val: images/val                      # Validation set images path
test: images/test                    # Test set images path

# Class information
nc: 2                                # Number of classes
names: ['Standing', 'Fall']           # Class names (corresponding to label IDs)
```

## 6. Model Training

Using the Python API of the Ultralytics framework, you can concisely and efficiently train a campus fall detection model with the following steps:

### 6.1 Pre-training Preparation

1. Ensure the dataset configuration file is correct
2. Prepare model configuration files or pre-trained weights:
   - Model configuration file (e.g., `yolov12-fall-detection.yaml`): Defines the model structure, can be modified based on framework default configurations;
   - Pre-trained weights (e.g., `yolov12n.pt`).

### 6.2 Training Code Example (Python API)

```python
# Import YOLO model
from ultralytics import YOLO

# Load the model (choose one of the two methods)
# Method 1: Build a new model from configuration file (training from scratch)
model = YOLO("yolov12-fall-detection.yaml")
# Method 2: Load a pre-trained model (recommended, fine-tuning based on existing weights)
model = YOLO("yolov12-fall-detection.pt") 
# Or load custom trained weights (for continued training)
# model = YOLO("runs/detect/campus_fall_train/weights/last.pt")

# Start training
results = model.train(
    data="ultralytics/cfg/datasets/campus_fall.yaml",  # Path to dataset configuration file
    epochs=200,                                        # Number of training epochs
    imgsz=640,                                         # Input image size
    batch=16,                                          # Batch size (adjust based on GPU memory)
    device=0,                                          # Training device (0=GPU, cpu=CPU)
    name="campus_fall_train",                          # Training task name (for result storage)
    plots=True                                         # Generate training visualization charts
)
```

### 6.3 Training Parameter Explanation

Core training parameters can be adjusted according to needs, with commonly used parameters as follows:

| Parameter Name | Description                                                  | Recommended Range              |
| -------------- | ------------------------------------------------------------ | ------------------------------ |
| `data`         | Path to dataset configuration file (.yaml)                   | Custom path                    |
| `epochs`       | Number of training epochs                                    | 100-300                        |
| `imgsz`        | Input image size (width = height)                            | 640/800/1024                   |
| `batch`        | Batch size (number of samples per training iteration)        | 8-32 (depending on GPU memory) |
| `device`       | Training device (`0` for first GPU, `cpu` for CPU)           | GPU preferred                  |
| `name`         | Training task name (results saved in `runs/detect/[name]/` directory) | Custom name                    |
| `lr0`          | Initial learning rate                                        | 0.001-0.01                     |
| `weight_decay` | Weight decay (to prevent overfitting)                        | 0.0005                         |
| `augment`      | Whether to enable advanced data augmentation                 | `True`                         |

### 6.4 Training Process Monitoring

Training logs and results are saved by default in the `ultralytics/runs/detect` directory, with core files including:

- `weights/`: Trained weights (`best.pt` is the best model on the validation set, `last.pt` is the model from the last epoch);
- `results.csv`: Training metrics record (loss, mAP@0.5, Precision, Recall, etc.);
- `confusion_matrix.png`: Confusion matrix (evaluating classification accuracy);
- `train_batch0.jpg`: Visualization of training batches (comparison of annotated and predicted boxes).

### 6.5 Model Validation and Testing

After training, you can evaluate model performance through the Python API:

```python
# Validate the model (using best weights to evaluate on validation set)
metrics = model.val(
    data="ultralytics/cfg/datasets/campus_fall.yaml",
    device=0
)
# Output key metrics
print(f"mAP@0.5: {metrics.box.map50:.3f}")  # mAP at 0.5 IoU
print(f"Precision: {metrics.box.p:.3f}")    # Precision
print(f"Recall: {metrics.box.r:.3f}")       # Recall

# Test the model (predict on test set images and save results)
results = model.predict(
    source="CampusFallDetectionDataset/images/test",  # Test set path
    save=True,                                         # Save predicted result images
    device=0,
    conf=0.5                                           # Confidence threshold (filter low-confidence predictions)
)
```

## 7. Model Deployment

The project provides multiple deployment options that can be selected according to actual scenarios, with core references in the `ultralytics/examples/` directory:

| Deployment Solution   | Application Scenario                       | Reference Example Directory            |
| --------------------- | ------------------------------------------ | -------------------------------------- |
| Python + ONNX Runtime | Quick verification, lightweight deployment | examples/YOLOv8-ONNXRuntime            |
| C++ Inference         | High-performance, low-latency scenarios    | examples/YOLOv8-CPP-Inference          |
| Rust Inference        | Cross-platform, high-security scenarios    | examples/YOLOv8-ONNXRuntime-Rust       |
| OpenVINO Deployment   | Intel CPU/GPU acceleration                 | examples/YOLOv8-OpenVINO-CPP-Inference |
| TFLite Deployment     | Mobile devices, embedded systems           | examples/YOLOv8-TFLite-Python          |

**ONNX Model Export (format conversion required before deployment)**:

```python
# Export to ONNX format (good cross-platform compatibility)
success = model.export(
    format="onnx",  # Export format
    imgsz=640       # Consistent with training input size
)
```

## 8. Customization and Extension

This project supports flexible customization and function extension to adapt to different needs, with core extension directions as follows:

### 8.1 Dataset Extension

- **Add new scenario data**: To cover more campus scenarios (such as nighttime corridors), supplement images and labels according to the existing directory structure. Simply place new data in `images/train` (or val/test) and `labels/train` (or val/test) without modifying configuration files;

- Add category annotations

  : To detect behaviors other than "fall/normal" (such as "running" or "climbing"):

  1. Add new category IDs in label files (e.g., `2: run`);
  2. Update `nc` (number of classes) and `names` (class name list) in `campus_fall.yaml`;
  3. Retrain the model to adapt to new categories.

### 8.2 Model and Training Customization

- Adjust training strategy: Add parameters in `model.train()`, for example:

  ```python
  model.train(
      # ... other parameters
      optimizer="Adam",  # Use Adam optimizer (default is SGD)
      lr0=0.001,         # Initial learning rate
      augment=True,      # Enable advanced data augmentation
      patience=30        # Early stopping strategy (stop if no improvement for 30 epohs)
  )
  ```

- **Custom loss function**: To optimize detection accuracy for fall behavior (e.g., addressing class imbalance), modify or add loss calculation logic in `ultralytics/nn/modules/loss.py` and reference it in the model configuration file.

### 8.3 Function Extension

- Add real-time warning module: Combine OpenCV to read camera video streams and trigger warnings when "fall" behavior is detected (such as pop-ups, sound prompts, message pushes). Core code example:

  ```python
  from ultralytics import YOLO
  import cv2
  
  model = YOLO("runs/detect/campus_fall_train/weights/best.pt")
  cap = cv2.VideoCapture(0)  # 0 represents local camera
  
  while cap.isOpened():
      ret, frame = cap.read()
      if not ret: break
      
      # Model inference
      results = model(frame, conf=0.5)  # conf=0.5 filters low-confidence results
      pred_classes = [result.boxes.cls.numpy() for result in results]
      
      # Trigger warning when fall behavior is detected
      if any(1 in cls for cls in pred_classes):
          cv2.putText(frame, "FALL DETECTED!", (50, 50), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
      
      # Display results
      cv2.imshow("Campus Fall Detection", results[0].plot())
      if cv2.waitKey(1) & 0xFF == ord('q'): break
  
  cap.release()
  cv2.destroyAllWindows()
  ```

## 9. Frequently Asked Questions (FAQ)

1. **GPU memory 不足 during training?**

   Reduce the `batch` size (e.g., from 16 to 8), decrease `imgsz` (e.g., from 800 to 640), or use a smaller model (e.g., yolov12n instead of yolov12x).

2. **High miss rate for fall behavior detection?**

   - Increase training data for fall scenarios (especially low-light and occluded scenes);
   - Lower the `conf` parameter during inference (e.g., `conf=0.3`) to retain more candidate boxes;
   - Extend training epochs or adjust learning rate strategy (e.g., `lr0=0.0005`).

3. **How to deploy the model to embedded devices?**

   Export to TFLite or ONNX format, use an inference framework supported by the device (such as TensorFlow Lite, ONNX Runtime Mobile), and enable model quantization (`export int8=True`) to reduce model size.

## 10. Acknowledgements

- Thanks to the Ultralytics team for providing the YOLO framework, which offers efficient and user-friendly tools for object detection tasks;
- Thanks to all individuals who provided support for annotating the campus fall detection dataset.
