"""
Single Face Classification Module

This module provides functionality for classifying individual face images using
a pre-trained FaceNet model and a trained classifier. It's designed for inference
on single images to identify which person the face belongs to.

License: MIT

Model Files:
    - FaceNet model: Pre-trained frozen graph (.pb) file containing the neural network
    - Classifier: Pickled file containing trained classifier and class names list
"""

import argparse
import pickle
import sys
import os
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python.platform import gfile
import cv2

# Disable TensorFlow v2 behavior to ensure compatibility with v1 model format
tf.disable_v2_behavior()


def load_model(model_path):
    """
    Load a frozen FaceNet model from a protobuf (.pb) file.
    
    This function loads a pre-trained FaceNet model that has been frozen
    (converted from checkpoint format to a single .pb file). The model
    is imported into the default TensorFlow graph for inference.
    
    Args:
        model_path (str): Path to the frozen model file (.pb format)
    
    Raises:
        SystemExit: If the model file doesn't exist or cannot be loaded
    
    Note:
        - Uses TensorFlow 1.x compatibility mode
        - The model is loaded into the default graph with no name prefix
        - Expected tensor names: 'input:0', 'embeddings:0', 'phase_train:0'
    """
    # Expand user path (handles ~ notation)
    model_exp = os.path.expanduser(model_path)
    
    if os.path.isfile(model_exp):
        print(f'Loading FaceNet model: {model_exp}')
        
        # Load the frozen graph definition from the .pb file
        with gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            
            # Import the graph definition into the default graph
            # name='' means no prefix is added to tensor names
            tf.import_graph_def(graph_def, name='')
            
        print('Model loaded successfully')
    else:
        print(f'ERROR: Model file not found: {model_exp}')
        print('Please ensure the FaceNet model (.pb file) exists at the specified path')
        sys.exit(-1)


def load_classifier(classifier_path):
    """
    Load a trained classifier and associated class names from a pickle file.
    
    The classifier file should contain a tuple of (trained_model, class_names)
    where the trained model is typically an SVM or similar classifier trained
    on FaceNet embeddings, and class_names is a list of person names/labels.
    
    Args:
        classifier_path (str): Path to the pickled classifier file (.pkl format)
    
    Returns:
        tuple: (classifier_model, class_names) where:
            - classifier_model: Trained sklearn classifier (e.g., SVM)
            - class_names: List of class names corresponding to classifier output indices
    
    Raises:
        SystemExit: If the classifier file doesn't exist or cannot be loaded
    
    Note:
        - Expects the pickle file to contain exactly 2 objects: (model, class_names)
        - The classifier should have a predict_proba method for confidence scores
    """
    if not os.path.exists(classifier_path):
        print(f'ERROR: Classifier file not found: {classifier_path}')
        print('Please ensure the trained classifier (.pkl file) exists at the specified path')
        sys.exit(-1)
    
    try:
        # Load the pickled classifier and class names
        with open(classifier_path, 'rb') as f:
            model, class_names = pickle.load(f)
        
        print(f'Loaded classifier with {len(class_names)} classes')
        print(f'Classes: {", ".join(class_names[:5])}{"..." if len(class_names) > 5 else ""}')
        
        return model, class_names
        
    except Exception as e:
        print(f'ERROR: Failed to load classifier: {str(e)}')
        print('Please ensure the classifier file is valid and contains (model, class_names)')
        sys.exit(-1)


def preprocess_image(image_path, image_size=160):
    """
    Load and preprocess a single image for FaceNet inference.
    
    This function handles the complete preprocessing pipeline required for FaceNet:
    1. Load image from disk
    2. Convert color space from BGR to RGB
    3. Resize to the required input size (typically 160x160)
    4. Normalize pixel values to [-1, 1] range (FaceNet requirement)
    5. Add batch dimension for TensorFlow input
    
    Args:
        image_path (str): Path to the input image file
        image_size (int, optional): Target size for the square image. Defaults to 160.
    
    Returns:
        numpy.ndarray or None: Preprocessed image array with shape (1, image_size, image_size, 3)
                              or None if image loading/processing fails
    
    Note:
        - Assumes input image contains a properly cropped and aligned face
        - Uses the same normalization as used during FaceNet training
        - Output is ready for direct input to FaceNet model
    """
    # Check if image file exists
    if not os.path.exists(image_path):
        print(f'ERROR: Image file not found: {image_path}')
        return None
    
    try:
        # Load image using OpenCV (loads in BGR format)
        img = cv2.imread(image_path)
        if img is None:
            print(f'ERROR: Could not load image: {image_path}')
            print('Please ensure the image file is valid and in a supported format')
            return None
        
        # Convert from BGR (OpenCV default) to RGB (FaceNet expects RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image to the required input size (maintains aspect ratio may be lost)
        # FaceNet typically expects 160x160 pixel images
        img = cv2.resize(img, (image_size, image_size))
        
        # Normalize pixel values to [-1, 1] range as expected by FaceNet
        # Original range [0, 255] -> subtract 127.5 -> [-127.5, 127.5] -> divide by 128 -> [-1, 1]
        img = (img.astype(np.float32) - 127.5) / 128.0
        
        # Add batch dimension: (height, width, channels) -> (1, height, width, channels)
        # TensorFlow expects batch dimension even for single images
        img = np.expand_dims(img, axis=0)
        
        print(f'Image preprocessed successfully: {img.shape}')
        return img
        
    except Exception as e:
        print(f'ERROR: Failed to preprocess image {image_path}: {str(e)}')
        return None


def get_embedding(sess, image, images_placeholder, embeddings, phase_train_placeholder):
    """
    Extract face embedding from preprocessed image using loaded FaceNet model.
    
    This function runs the preprocessed image through the FaceNet model to
    generate a high-dimensional embedding vector that represents the face.
    The embedding can then be used for face recognition and classification.
    
    Args:
        sess (tf.Session): Active TensorFlow session with loaded model
        image (numpy.ndarray): Preprocessed image array with batch dimension
        images_placeholder (tf.Tensor): Input tensor placeholder for images
        embeddings (tf.Tensor): Output tensor for face embeddings
        phase_train_placeholder (tf.Tensor): Training phase placeholder (should be False for inference)
    
    Returns:
        numpy.ndarray: Face embedding vector, typically 128 or 512 dimensions
    
    Note:
        - phase_train_placeholder must be set to False for inference
        - The embedding is a dense vector representation of the face
        - Similar faces should have similar embeddings (small Euclidean distance)
    """
    # Create feed dictionary for the TensorFlow session
    feed_dict = {
        images_placeholder: image,           # Input image batch
        phase_train_placeholder: False       # Set to False for inference (no dropout, etc.)
    }
    
    # Run the model to get face embedding
    embedding = sess.run(embeddings, feed_dict=feed_dict)
    
    print(f'Generated embedding with shape: {embedding.shape}')
    return embedding


def classify_face(image_path, model_path, classifier_path, top_k=3):
    """
    Complete pipeline to classify a single face image.
    
    This is the main function that orchestrates the entire classification process:
    1. Load the trained classifier
    2. Preprocess the input image
    3. Load the FaceNet model
    4. Extract face embedding
    5. Classify the embedding
    6. Return top-K predictions with confidence scores
    
    Args:
        image_path (str): Path to the face image to classify
        model_path (str): Path to the FaceNet model (.pb file)
        classifier_path (str): Path to the trained classifier (.pkl file)
        top_k (int, optional): Number of top predictions to return. Defaults to 3.
    
    Returns:
        tuple or None: (best_class_name, best_confidence) if successful, None if failed
    
    Example:
        >>> result = classify_face('face.jpg', 'facenet.pb', 'classifier.pkl')
        >>> if result:
        >>>     name, confidence = result
        >>>     print(f"Predicted: {name} with {confidence:.3f} confidence")
    """
    print(f'Starting face classification for: {image_path}')
    print('=' * 60)
    
    # Step 1: Load the trained classifier and class names
    print('Step 1: Loading classifier...')
    classifier, class_names = load_classifier(classifier_path)
    
    # Step 2: Preprocess the input image
    print('Step 2: Preprocessing image...')
    image = preprocess_image(image_path)
    if image is None:
        print('Classification failed: Could not preprocess image')
        return None
    
    # Step 3: Create TensorFlow session and load model
    print('Step 3: Loading FaceNet model and creating session...')
    with tf.Session() as sess:
        # Load the FaceNet model into the current session
        load_model(model_path)
        
        # Get references to the required tensors from the loaded graph
        try:
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        except KeyError as e:
            print(f'ERROR: Required tensor not found in model: {e}')
            print('Please ensure you are using a compatible FaceNet model')
            return None
        
        # Step 4: Extract face embedding
        print('Step 4: Extracting face embedding...')
        embedding = get_embedding(sess, image, images_placeholder, embeddings, phase_train_placeholder)
        
        # Step 5: Classify the embedding
        print('Step 5: Running classification...')
        try:
            # Get prediction probabilities for all classes
            predictions = classifier.predict_proba(embedding)[0]
            
            # Get indices of top-k predictions (sorted by confidence, highest first)
            top_indices = np.argsort(predictions)[::-1][:top_k]
            
            # Display results
            print(f'\nClassification results for: {os.path.basename(image_path)}')
            print('-' * 50)
            
            for i, idx in enumerate(top_indices):
                confidence = predictions[idx]
                person_name = class_names[idx]
                print(f'{i+1}. {person_name}: {confidence:.3f} ({confidence*100:.1f}%)')
            
            # Get the best prediction
            best_class = class_names[top_indices[0]]
            best_confidence = predictions[top_indices[0]]
            
            print(f'\nBest match: {best_class} (confidence: {best_confidence:.3f})')
            print('=' * 60)
            
            return best_class, best_confidence
            
        except Exception as e:
            print(f'ERROR: Classification failed: {str(e)}')
            return None


def main():
    """
    Command-line interface for single face classification.
    
    This function provides a command-line interface that allows users to
    classify face images from the terminal with various options.
    
    Command-line Arguments:
        --image-path: Path to the face image to classify (required)
        --model-path: Path to the FaceNet model .pb file (required)
        --classifier-path: Path to the trained classifier .pkl file (required)
        --top-k: Number of top predictions to display (optional, default: 3)
    
    """
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(
        description='Classify a single face image using FaceNet and trained classifier',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Requirements:
  - FaceNet model: Pre-trained frozen graph (.pb file)
  - Classifier: Trained classifier with class names (.pkl file)
  - Input image: Preprocessed face image (preferably aligned and cropped)
        """
    )
    
    # Define required arguments
    parser.add_argument('--image-path', type=str, required=True,
                        help='Path to the face image to classify')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the FaceNet model (.pb file)')
    parser.add_argument('--classifier-path', type=str, required=True,
                        help='Path to the trained classifier (.pkl file)')
    
    # Define optional arguments
    parser.add_argument('--top-k', type=int, default=3,
                        help='Number of top predictions to show (default: 3)')
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # Validate arguments
    if args.top_k <= 0:
        print('ERROR: top-k must be a positive integer')
        sys.exit(-1)
    
    # Log the configuration
    print('Face Classification Configuration:')
    print(f'  Image path: {args.image_path}')
    print(f'  Model path: {args.model_path}')
    print(f'  Classifier path: {args.classifier_path}')
    print(f'  Top-K predictions: {args.top_k}')
    print()
    
    # Run the classification
    result = classify_face(args.image_path, args.model_path, args.classifier_path, args.top_k)
    
    # Handle the result
    if result is None:
        print('Classification failed. Please check the error messages above.')
        sys.exit(-1)
    else:
        print('Classification completed successfully!')


if __name__ == '__main__':
    main()