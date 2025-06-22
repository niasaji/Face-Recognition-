## FaceNet Face Recognition System
A comprehensive face recognition pipeline using FaceNet embeddings and SVM classification. This project provides a complete toolkit for face detection, alignment, preprocessing, training classifiers, and performing face recognition with high accuracy using dlib and TensorFlow.

### Features
Face Detection & Alignment: High-quality face alignment using dlib's 68-point facial landmark detection  
Parallel Processing: Multi-core preprocessing for efficient batch operations  
FaceNet Integration: Leverages pre-trained FaceNet models for high-quality face embeddings  
SVM Classification: Support Vector Machine classifier for person identification  
Single Image Inference: Real-time classification of individual face images  
Data Augmentation: Built-in image augmentation including random flipping, brightness, and contrast adjustment  
Dataset Management: Utilities for loading, filtering, and splitting face recognition datasets  
Batch Processing: Efficient TensorFlow input pipelines for training and inference

### Requirements

Python 3.6+  
OpenCV  
dlib  
TensorFlow 1.x (recommended: 1.15.0) or TensorFlow 2.x with compatibility mode  
NumPy  
scikit-learn  
Requests

FaceNet Model: Pre-trained frozen graph file (.pb format)

###### Facial Landmarks Predictor:

Download shape_predictor_68_face_landmarks.dat from dlib models  
Place in the same directory as your scripts

#### Training Options:
--model-path: Path to FaceNet model (.pb file)  
--input-dir: Directory with processed face images  
--classifier-path: Output path for trained classifier  
--is-train: Enable training mode  
--min-num-images-per-class: Minimum images required per person  
--batch-size: Batch size for processing (default: 128)  
--num-epochs: Number of training epochs (default: 3)  
--split-ratio: Train/test split ratio (default: 0.7)

#### Classification Options:
--image-path: Path to face image to classify  
--model-path: Path to FaceNet model  
--classifier-path: Path to trained classifier  
--top-k: Number of top predictions to show (default: 3)

### Core Components
Face Alignment (align_dlib.py)  
The AlignDlib class provides robust face alignment functionality:

Face Detection: Uses dlib's HOG-based face detector  
Landmark Detection: Identifies 68 facial landmarks  
Alignment: Transforms faces to a standard pose using affine transformations  
Cropping: Outputs consistently sized face images

###  Configuration Options

#####Face Alignment Parameters

imgDim: Output image dimensions (square)  
landmarkIndices: Which landmarks to use for alignment

INNER_EYES_AND_BOTTOM_LIP (default): More stable alignment  
OUTER_EYES_AND_NOSE: Alternative alignment strategy

scale: Scale factor for aligned faces  
skipMulti: Skip images with multiple faces

##### Data Augmentation Options

random_flip: Random horizontal flipping  
random_brightness: Brightness adjustment (±0.3)  
random_contrast: Contrast adjustment (0.2-1.8x)  
Per-image standardization (automatic)
