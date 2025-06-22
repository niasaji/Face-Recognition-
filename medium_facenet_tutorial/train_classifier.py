"""
Face Recognition Classifier Training Script

This script trains a Support Vector Machine (SVM) classifier on face embeddings generated 
from a pre-trained FaceNet model. It can be used to train a classifier to recognize 
specific individuals based on their facial features.

Original code adapted from OpenFace project:
https://github.com/cmusatyalab/openface/blob/master/openface/train_classifier.py
License : MIT
"""

import argparse
import logging
import os
import pickle
import sys
import time

import numpy as np

# Using TensorFlow 1.x compatibility mode for legacy FaceNet models
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from sklearn.svm import SVC
from tensorflow.python.platform import gfile

# Import custom modules for data handling
from lfw_input import filter_dataset, split_dataset, get_dataset
from medium_facenet_tutorial import lfw_input

# Set up logging
logger = logging.getLogger(__name__)


def main(input_directory, model_path, classifier_output_path, batch_size, num_threads, num_epochs,
         min_images_per_labels, split_ratio, is_train=True):
    """
    Main function that orchestrates the training/evaluation pipeline.
    
    Loads face images from the input directory, generates embeddings using a pre-trained
    FaceNet model, and either trains a new SVM classifier or evaluates an existing one.
    
    Args:
        input_directory (str): Path to directory containing pre-processed face images
                              organized in subdirectories (one per person)
        model_path (str): Path to frozen FaceNet protobuf graph file (.pb)
        classifier_output_path (str): Path where trained classifier will be saved/loaded
        batch_size (int): Number of images to process in each batch
        num_threads (int): Number of threads for data loading pipeline
        num_epochs (int): Number of times to iterate through the dataset
        min_images_per_labels (int): Minimum number of images required per person/class
        split_ratio (float): Ratio for train/test split (e.g., 0.7 = 70% train, 30% test)
        is_train (bool): True for training mode, False for evaluation mode
    """
    start_time = time.time()
    
    # Create TensorFlow session with optimized configuration
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        
        # Load and split the dataset into training and testing sets
        train_set, test_set = _get_test_and_train_set(
            input_directory, 
            min_num_images_per_label=min_images_per_labels,
            split_ratio=split_ratio
        )
        
        # Load images and labels based on mode (train vs evaluate)
        if is_train:
            # Training mode: use data augmentation for better generalization
            images, labels, class_names = _load_images_and_labels(
                train_set, 
                image_size=160,  # FaceNet expects 160x160 input images
                batch_size=batch_size,
                num_threads=num_threads, 
                num_epochs=num_epochs,
                random_flip=True,      # Horizontal flip augmentation
                random_brightness=True, # Brightness variation augmentation
                random_contrast=True   # Contrast variation augmentation
            )
        else:
            # Evaluation mode: no augmentation, single epoch
            images, labels, class_names = _load_images_and_labels(
                test_set, 
                image_size=160, 
                batch_size=batch_size,
                num_threads=num_threads, 
                num_epochs=1  # Only one pass through test data
            )

        # Load the pre-trained FaceNet model
        _load_model(model_filepath=model_path)

        # Initialize TensorFlow variables
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        # Get references to the model's input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embedding_layer = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        # Set up TensorFlow queue coordinator for efficient data loading
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        # Generate embeddings for all images using the FaceNet model
        emb_array, label_array = _create_embeddings(
            embedding_layer, images, labels, images_placeholder,
            phase_train_placeholder, sess
        )

        # Clean up TensorFlow threads
        coord.request_stop()
        coord.join(threads=threads)
        logger.info('Created {} embeddings'.format(len(emb_array)))

        classifier_filename = classifier_output_path

        if is_train:
            # Training mode: train and save new classifier
            _train_and_save_classifier(emb_array, label_array, class_names, classifier_filename)
        else:
            # Evaluation mode: load and evaluate existing classifier
            _evaluate_classifier(emb_array, label_array, classifier_filename)

        logger.info('Completed in {} seconds'.format(time.time() - start_time))


def _get_test_and_train_set(input_dir, min_num_images_per_label, split_ratio=0.7):
    """
    Load dataset and split into training and testing sets.
    
    Filters out classes with insufficient images and splits the remaining data
    according to the specified ratio.
    
    Args:
        input_dir (str): Directory containing face images organized by person
        min_num_images_per_label (int): Minimum images required per person
        split_ratio (float): Fraction of data to use for training (0.0-1.0)
        
    Returns:
        tuple: (train_set, test_set) - Lists of image paths and labels
    """
    # Load all images from the input directory
    dataset = get_dataset(input_dir)
    
    # Remove classes with insufficient training examples
    dataset = filter_dataset(dataset, min_images_per_label=min_num_images_per_label)
    
    # Split into training and testing sets
    train_set, test_set = split_dataset(dataset, split_ratio=split_ratio)

    return train_set, test_set


def _load_images_and_labels(dataset, image_size, batch_size, num_threads, num_epochs, 
                           random_flip=False, random_brightness=False, random_contrast=False):
    """
    Create TensorFlow data pipeline for loading and preprocessing images.
    
    Sets up efficient batched loading of images with optional data augmentation
    for training robustness.
    
    Args:
        dataset: Dataset object containing image paths and labels
        image_size (int): Target size for square images (e.g., 160 for 160x160)
        batch_size (int): Number of images per batch
        num_threads (int): Number of parallel threads for data loading
        num_epochs (int): Number of times to iterate through the dataset
        random_flip (bool): Whether to randomly flip images horizontally
        random_brightness (bool): Whether to randomly adjust brightness
        random_contrast (bool): Whether to randomly adjust contrast
        
    Returns:
        tuple: (images_tensor, labels_tensor, class_names_list)
    """
    # Extract class names for later use in classifier
    class_names = [cls.name for cls in dataset]
    
    # Get all image paths and corresponding numeric labels
    image_paths, labels = lfw_input.get_image_paths_and_labels(dataset)
    
    # Create TensorFlow data pipeline with preprocessing
    images, labels = lfw_input.read_data(
        image_paths, labels, image_size, batch_size, num_epochs, num_threads,
        shuffle=False,  # Shuffling handled by queue
        random_flip=random_flip,
        random_brightness=random_brightness,
        random_contrast=random_contrast
    )
    
    return images, labels, class_names


def _load_model(model_filepath):
    """
    Load a frozen TensorFlow protobuf graph file.
    
    The FaceNet model should be a frozen graph (.pb file) that can be loaded
    directly without requiring checkpoint files.
    
    Args:
        model_filepath (str): Path to the frozen protobuf graph file
        
    Raises:
        SystemExit: If the model file is not found
    """
    # Expand user path (handles ~/path notation)
    model_exp = os.path.expanduser(model_filepath)
    
    if os.path.isfile(model_exp):
        logging.info('Model filename: %s' % model_exp)
        
        # Load and parse the protobuf graph
        with gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            # Import the graph into the current TensorFlow session
            tf.import_graph_def(graph_def, name='')
    else:
        logger.error('Missing model file. Exiting')
        sys.exit(-1)


def _create_embeddings(embedding_layer, images, labels, images_placeholder, 
                      phase_train_placeholder, sess):
    """
    Generate face embeddings using the loaded FaceNet model.
    
    Processes images in batches through the neural network to create
    high-dimensional feature vectors (embeddings) that represent each face.
    
    Args:
        embedding_layer: TensorFlow tensor representing the embedding output
        images: TensorFlow tensor for input image batches
        labels: TensorFlow tensor for corresponding labels
        images_placeholder: Input placeholder for feeding images to the model
        phase_train_placeholder: Placeholder for training phase (set to False for inference)
        sess: Active TensorFlow session
        
    Returns:
        tuple: (embeddings_array, labels_array) - NumPy arrays of embeddings and labels
    """
    emb_array = None
    label_array = None
    
    try:
        i = 0
        # Process all batches until queue is exhausted
        while True:
            # Get next batch of images and labels from the queue
            batch_images, batch_labels = sess.run([images, labels])
            logger.info('Processing iteration {} batch of size: {}'.format(i, len(batch_labels)))
            
            # Run inference to get embeddings for this batch
            emb = sess.run(embedding_layer,
                          feed_dict={
                              images_placeholder: batch_images, 
                              phase_train_placeholder: False  # Set to inference mode
                          })

            # Accumulate embeddings and labels
            emb_array = np.concatenate([emb_array, emb]) if emb_array is not None else emb
            label_array = np.concatenate([label_array, batch_labels]) if label_array is not None else batch_labels
            i += 1

    except tf.errors.OutOfRangeError:
        # This exception is expected when all data has been processed
        pass

    return emb_array, label_array


def _train_and_save_classifier(emb_array, label_array, class_names, classifier_filename_exp):
    """
    Train an SVM classifier on the face embeddings and save it to disk.
    
    Uses a linear SVM with probability estimates enabled for confidence scores.
    The trained model and class names are pickled together for later use.
    
    Args:
        emb_array (np.ndarray): Face embeddings array (n_samples, embedding_dim)
        label_array (np.ndarray): Corresponding labels array (n_samples,)
        class_names (list): List of class names corresponding to label indices
        classifier_filename_exp (str): Path where the trained classifier will be saved
    """
    logger.info('Training Classifier')
    
    # Initialize SVM classifier
    # Linear kernel works well with high-dimensional embeddings from deep networks
    # Probability=True enables predict_proba() for confidence scores
    model = SVC(kernel='linear', probability=True, verbose=False)
    
    # Train the classifier on the embeddings
    model.fit(emb_array, label_array)

    # Save both the trained model and class names for later use
    with open(classifier_filename_exp, 'wb') as outfile:
        pickle.dump((model, class_names), outfile)
    
    logging.info('Saved classifier model to file "%s"' % classifier_filename_exp)


def _evaluate_classifier(emb_array, label_array, classifier_filename):
    """
    Load a trained classifier and evaluate it on the provided embeddings.
    
    Prints prediction results and overall accuracy to help assess model performance.
    
    Args:
        emb_array (np.ndarray): Face embeddings for evaluation
        label_array (np.ndarray): True labels for the embeddings
        classifier_filename (str): Path to the saved classifier pickle file
        
    Raises:
        ValueError: If the classifier file doesn't exist
    """
    logger.info('Evaluating classifier on {} images'.format(len(emb_array)))
    
    # Check if classifier file exists
    if not os.path.exists(classifier_filename):
        raise ValueError('Pickled classifier not found, have you trained first?')

    # Load the trained classifier and class names
    with open(classifier_filename, 'rb') as f:
        model, class_names = pickle.load(f)

        # Get prediction probabilities for all samples
        predictions = model.predict_proba(emb_array)
        
        # Find the class with highest probability for each sample
        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

        # Print individual predictions with confidence scores
        for i in range(len(best_class_indices)):
            print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))

        # Calculate and print overall accuracy
        accuracy = np.mean(np.equal(best_class_indices, label_array))
        print('Accuracy: %.3f' % accuracy)


if __name__ == '__main__':
    # Set up logging to display info messages
    logging.basicConfig(level=logging.INFO)
    
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description='Train or evaluate a face recognition classifier using FaceNet embeddings',
        add_help=True
    )
    
    # Required arguments
    parser.add_argument('--model-path', type=str, action='store', dest='model_path',
                        required=True, help='Path to FaceNet model protobuf graph (.pb file)')
    parser.add_argument('--input-dir', type=str, action='store', dest='input_dir',
                        required=True, help='Input directory containing face images organized by person')
    parser.add_argument('--classifier-path', type=str, action='store', dest='classifier_path',
                        required=True, help='Path to save/load the trained classifier model')
    
    # Optional arguments with defaults
    parser.add_argument('--batch-size', type=int, action='store', dest='batch_size',
                        default=128, help='Batch size for processing images (default: 128)')
    parser.add_argument('--num-threads', type=int, action='store', dest='num_threads', 
                        default=16, help='Number of threads for data loading (default: 16)')
    parser.add_argument('--num-epochs', type=int, action='store', dest='num_epochs', 
                        default=3, help='Number of epochs to train (default: 3)')
    parser.add_argument('--split-ratio', type=float, action='store', dest='split_ratio', 
                        default=0.7, help='Train/test split ratio (default: 0.7)')
    parser.add_argument('--min-num-images-per-class', type=int, action='store', default=10,
                        dest='min_images_per_class', 
                        help='Minimum number of images required per person (default: 10)')
    
    # Mode selection
    parser.add_argument('--is-train', action='store_true', dest='is_train', default=False,
                        help='Use training mode (default: evaluation mode)')

    # Parse command-line arguments
    args = parser.parse_args()

    # Run the main function with parsed arguments
    main(input_directory=args.input_dir, 
         model_path=args.model_path, 
         classifier_output_path=args.classifier_path,
         batch_size=args.batch_size, 
         num_threads=args.num_threads, 
         num_epochs=args.num_epochs,
         min_images_per_labels=args.min_images_per_class, 
         split_ratio=args.split_ratio, 
         is_train=args.is_train)