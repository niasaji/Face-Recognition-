"""
LFW Dataset Input Pipeline Module

This module provides utilities for loading and preprocessing image datasets,
particularly designed for face recognition tasks using the Labeled Faces in the Wild (LFW) dataset.
It creates TensorFlow input pipelines with data augmentation capabilities.

Original code adapted from OpenFace project:
https://github.com/cmusatyalab/openface/blob/master/openface/lfw_input.py
License : MIT

"""

import logging
import os

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import ops

# Configure module logger
logger = logging.getLogger(__name__)


def read_data(image_paths, label_list, image_size, batch_size, max_nrof_epochs, num_threads, shuffle, random_flip,
              random_brightness, random_contrast):
    """
    Creates TensorFlow input queue for batch loading images with data augmentation.
    
    This function sets up a TensorFlow input pipeline that reads images from disk,
    applies various data augmentation techniques, and batches them for training.
    Uses TensorFlow 1.x compatibility mode for legacy support.
    
    Args:
        image_paths (list): List of file paths to image files
        label_list (list): List of integer labels corresponding to each image
        image_size (int): Target size to resize images to (assumes square images)
        batch_size (int): Number of images to include in each batch
        max_nrof_epochs (int): Maximum number of epochs to iterate through the dataset
        num_threads (int): Number of parallel threads for data loading
        shuffle (bool): Whether to shuffle the dataset
        random_flip (bool): Whether to apply random horizontal flipping
        random_brightness (bool): Whether to apply random brightness adjustment
        random_contrast (bool): Whether to apply random contrast adjustment
    
    Returns:
        tuple: (image_batch, label_batch) where:
            - image_batch: Tensor of shape [batch_size, image_size, image_size, 3]
            - label_batch: Tensor of shape [batch_size] containing integer labels
    
    Note:
        Images are automatically standardized using per-image standardization.
        Random crop is applied to extract image_size x image_size patches from loaded images.
    """
    # Convert Python lists to TensorFlow tensors
    images = ops.convert_to_tensor(image_paths, dtype=tf.string)
    labels = ops.convert_to_tensor(label_list, dtype=tf.int32)

    # Create input queue for feeding data - using tf.compat.v1 for backward compatibility
    input_queue = tf.compat.v1.train.slice_input_producer((images, labels),
                                                          num_epochs=max_nrof_epochs, 
                                                          shuffle=shuffle)

    # Prepare lists to collect processed images and labels from multiple threads
    images_labels = []
    imgs = []
    lbls = []
    
    # Process images using multiple threads for performance
    for _ in range(num_threads):
        # Read and decode image from disk
        image, label = read_image_from_disk(filename_to_label_tuple=input_queue)
        
        # Apply random crop to get consistent image size
        image = tf.random_crop(image, size=[image_size, image_size, 3])
        image.set_shape((image_size, image_size, 3))
        
        # Normalize image using per-image standardization (zero mean, unit variance)
        image = tf.image.per_image_standardization(image)

        # Apply data augmentation techniques based on configuration
        if random_flip:
            # Randomly flip image horizontally with 50% probability
            image = tf.image.random_flip_left_right(image)

        if random_brightness:
            # Adjust brightness randomly within [-0.3, 0.3] range
            image = tf.image.random_brightness(image, max_delta=0.3)

        if random_contrast:
            # Adjust contrast randomly within [0.2, 1.8] range
            image = tf.image.random_contrast(image, lower=0.2, upper=1.8)

        # Collect processed images and labels
        imgs.append(image)
        lbls.append(label)
        images_labels.append([image, label])

    # Create batches from processed images using multiple threads
    image_batch, label_batch = tf.compat.v1.train.batch_join(images_labels,
                                                             batch_size=batch_size,
                                                             capacity=4 * num_threads,
                                                             enqueue_many=False,
                                                             allow_smaller_final_batch=True)
    return image_batch, label_batch


def read_image_from_disk(filename_to_label_tuple):
    """
    Reads and decodes a JPEG image from disk given a filename-label tuple.
    
    This function is designed to work with TensorFlow's input queue system.
    It takes a tuple containing filename and label, reads the image file,
    and decodes it as a JPEG with 3 color channels (RGB).
    
    Args:
        filename_to_label_tuple (tuple): Tuple containing:
            - [0]: String tensor with image file path
            - [1]: Integer tensor with image label
    
    Returns:
        tuple: (decoded_image, label) where:
            - decoded_image: 3D tensor with shape [height, width, 3] representing RGB image
            - label: Integer tensor representing the image class label
    
    Note:
        Uses tf.io.read_file for TensorFlow 2.x compatibility
    """
    # Extract label from the input tuple
    label = filename_to_label_tuple[1]
    
    # Read the image file contents as binary data
    file_contents = tf.io.read_file(filename_to_label_tuple[0])  # Updated for TF 2.x compatibility
    
    # Decode JPEG image with 3 channels (RGB)
    example = tf.image.decode_jpeg(file_contents, channels=3)
    
    return example, label


def get_image_paths_and_labels(dataset):
    """
    Flattens a dataset of ImageClass objects into separate lists of image paths and labels.
    
    This utility function takes a structured dataset where each class contains
    multiple image paths and converts it into flat lists suitable for training.
    Each image gets assigned a label corresponding to its class index.
    
    Args:
        dataset (list): List of ImageClass objects, each containing image paths for one class
    
    Returns:
        tuple: (image_paths_flat, labels_flat) where:
            - image_paths_flat: List of all image file paths across all classes
            - labels_flat: List of integer labels corresponding to each image path
    
    """
    image_paths_flat = []
    labels_flat = []
    
    # Iterate through each class in the dataset
    for i in range(int(len(dataset))):
        # Add all image paths from current class
        image_paths_flat += dataset[i].image_paths
        # Assign class index as label for all images in this class
        labels_flat += [i] * len(dataset[i].image_paths)
        
    return image_paths_flat, labels_flat


def get_dataset(input_directory):
    """
    Loads a dataset from a directory structure where each subdirectory represents a class.
    
    Args:
        input_directory (str): Path to the root directory containing class subdirectories
    
    Returns:
        list: List of ImageClass objects, one for each class found in the directory
    
    Note:
        - Only processes subdirectories (ignores files in root directory)
        - Classes are sorted alphabetically by directory name
        - Each ImageClass contains the class name and list of image paths
    """
    dataset = []

    # Get list of all items in the input directory
    classes = os.listdir(input_directory)
    classes.sort()  # Sort alphabetically for consistent ordering
    nrof_classes = len(classes)
    
    # Process each class directory
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(input_directory, class_name)
        
        # Only process if it's actually a directory
        if os.path.isdir(facedir):
            # Get list of all images in this class directory
            images = os.listdir(facedir)
            # Create full paths to all images
            image_paths = [os.path.join(facedir, img) for img in images]
            # Create ImageClass object and add to dataset
            dataset.append(ImageClass(class_name, image_paths))

    return dataset


def filter_dataset(dataset, min_images_per_label=10):
    """
    Filters dataset to keep only classes with sufficient number of images.
    
    This function removes classes that don't have enough images for effective training.
    Classes with fewer than min_images_per_label images are excluded from the filtered dataset.
    
    Args:
        dataset (list): List of ImageClass objects to filter
        min_images_per_label (int, optional): Minimum number of images required per class.
                                            Defaults to 10.
    
    Returns:
        list: Filtered list of ImageClass objects containing only classes with
              sufficient number of images
    
    Note:
        - Logs information about skipped classes
        - Useful for ensuring all classes have enough data for train/test splits
    """
    filtered_dataset = []
    
    # Check each class in the dataset
    for i in range(len(dataset)):
        if len(dataset[i].image_paths) < min_images_per_label:
            # Log classes being skipped due to insufficient images
            logger.info('Skipping class: {} (only {} images, need {})'.format(
                dataset[i].name, len(dataset[i].image_paths), min_images_per_label))
            continue
        else:
            # Keep classes with sufficient images
            filtered_dataset.append(dataset[i])
            
    return filtered_dataset


def split_dataset(dataset, split_ratio=0.8):
    """
    Splits dataset into training and testing sets for each class.
    
    This function randomly splits each class into training and testing portions
    while maintaining the class structure. Each class is split independently
    according to the specified ratio.
    
    Args:
        dataset (list): List of ImageClass objects to split
        split_ratio (float, optional): Fraction of images to use for training.
                                     Defaults to 0.8 (80% train, 20% test).
    
    Returns:
        tuple: (train_set, test_set) where:
            - train_set: List of ImageClass objects for training
            - test_set: List of ImageClass objects for testing
    
    Note:
        - Classes with fewer than 2 images are skipped entirely
        - Images within each class are randomly shuffled before splitting
        - Maintains class structure in both training and testing sets
        - Uses split_ratio to determine train/test boundary (e.g., 0.8 = 80% train)
    """
    train_set = []
    test_set = []
    min_nrof_images = 2  # Minimum images needed to create both train and test sets
    
    # Process each class in the dataset
    for cls in dataset:
        paths = cls.image_paths
        
        # Randomly shuffle image paths to ensure random distribution
        np.random.shuffle(paths)
        
        # Calculate split point based on the specified ratio
        split = int(round(len(paths) * split_ratio))
        
        # Skip classes with insufficient images for both train and test sets
        if split < min_nrof_images:
            logger.warning('Skipping class {} - not enough images for train/test split '
                         '({} images, need at least {})'.format(cls.name, len(paths), min_nrof_images))
            continue
        
        # Create training set with first 'split' images
        train_set.append(ImageClass(cls.name, paths[0:split]))
        
        # Create test set with remaining images (note: [split:-1] excludes last image)
        # This might be intentional or could be a bug - consider using paths[split:] instead
        test_set.append(ImageClass(cls.name, paths[split:-1]))
    
    return train_set, test_set


class ImageClass:
    """
    Represents a class of images in a dataset.
    
    This class encapsulates information about a single class in an image dataset,
    including the class name and paths to all images belonging to that class.
    Commonly used in face recognition datasets where each class represents a person.
    
    Attributes:
        name (str): Name or identifier of the image class
        image_paths (list): List of file paths to images belonging to this class
    """
    
    def __init__(self, name, image_paths):
        """
        Initialize an ImageClass instance.
        
        Args:
            name (str): Name or identifier for this image class
            image_paths (list): List of file paths to images in this class
        """
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        #Return a human-readable string representation of the ImageClass.
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        #Return the number of images in this class.
        return len(self.image_paths)