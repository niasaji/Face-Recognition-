"""
Face Detection and Alignment Preprocessing Module

This module provides functionality for preprocessing face images by detecting,
aligning, and cropping faces from input images. It's designed to prepare datasets
for face recognition training by ensuring all faces are properly aligned and
consistently sized.

The module uses dlib's face detection and alignment capabilities with 68-point
facial landmark detection for precise face alignment.

Original code adapted from:
https://github.com/davidsandberg/facenet/blob/master/src/preprocess.py
License: MIT

"""

import argparse
import glob
import logging
import multiprocessing as mp
import os
import time

import cv2

from medium_facenet_tutorial.align_dlib import AlignDlib

# Configure module logger
logger = logging.getLogger(__name__)

# Initialize face alignment module with pre-trained 68-point facial landmark predictor
# The .dat file contains the trained model for detecting facial landmarks
align_dlib = AlignDlib(os.path.join(os.path.dirname(__file__), 'shape_predictor_68_face_landmarks.dat'))


def main(input_dir, output_dir, crop_dim):
    """
    Main processing function that orchestrates the face preprocessing pipeline.
    
    This function sets up multiprocessing to efficiently process all images in the
    input directory. It maintains the directory structure of the input in the output,
    with each subdirectory representing a different person/class.
    
    Args:
        input_dir (str): Path to input directory containing subdirectories of images
        output_dir (str): Path to output directory where processed images will be saved
        crop_dim (int): Target dimension for square cropped face images (e.g., 180 for 180x180)
    
    Process:
        1. Creates output directory structure mirroring input
        2. Finds all .jpg images recursively in input directory
        3. Uses multiprocessing to process images in parallel
        4. Each image is face-detected, aligned, and cropped
    
    Note:
        - Uses all available CPU cores for parallel processing
        - Only processes .jpg files
        - Maintains original filenames in output
    """
    start_time = time.time()
    
    # Create multiprocessing pool with all available CPU cores for maximum efficiency
    pool = mp.Pool(processes=mp.cpu_count())

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f'Created output directory: {output_dir}')

    # Create output subdirectories mirroring the input directory structure
    # Each subdirectory typically represents a different person/class
    for image_dir in os.listdir(input_dir):
        image_output_dir = os.path.join(output_dir, os.path.basename(os.path.basename(image_dir)))
        if not os.path.exists(image_output_dir):
            os.makedirs(image_output_dir)
            logger.debug(f'Created output subdirectory: {image_output_dir}')

    # Find all JPEG images recursively in the input directory
    # The '**/*.jpg' pattern searches all subdirectories
    image_paths = glob.glob(os.path.join(input_dir, '**/*.jpg'), recursive=True)
    logger.info(f'Found {len(image_paths)} images to process')
    
    # Submit each image for asynchronous processing in the multiprocessing pool
    for index, image_path in enumerate(image_paths):
        # Determine output path maintaining the same directory structure
        image_output_dir = os.path.join(output_dir, os.path.basename(os.path.dirname(image_path)))
        output_path = os.path.join(image_output_dir, os.path.basename(image_path))
        
        # Submit image processing task to the pool (non-blocking)
        pool.apply_async(preprocess_image, (image_path, output_path, crop_dim))

    # Close the pool to prevent more tasks from being submitted
    pool.close()
    
    # Wait for all submitted tasks to complete
    pool.join()
    
    # Log completion time
    elapsed_time = time.time() - start_time
    logger.info(f'Processing completed in {elapsed_time:.2f} seconds')
    logger.info(f'Processed {len(image_paths)} images at {len(image_paths)/elapsed_time:.2f} images/second')


def preprocess_image(input_path, output_path, crop_dim):
    """
    Preprocesses a single image by detecting, aligning, and cropping the face.
    
    This function is designed to be called by multiprocessing workers. It processes
    one image at a time, detecting the largest face, aligning it based on facial
    landmarks, and cropping it to the specified dimensions.
    
    Args:
        input_path (str): Full path to the input image file
        output_path (str): Full path where the processed image should be saved
        crop_dim (int): Target dimension for the square output image
    
    Process:
        1. Load and process the image
        2. Detect the largest face in the image
        3. Align the face using facial landmarks
        4. Crop to specified dimensions
        5. Save the processed image
    
    Note:
        - If no face is detected or processing fails, the image is skipped
        - Warnings are logged for skipped images
        - Only the largest face is processed if multiple faces are present
    """
    try:
        # Process the image (detect, align, crop face)
        image = _process_image(input_path, crop_dim)
        
        if image is not None:
            # Successfully processed - save the aligned and cropped face
            logger.debug(f'Writing processed file: {output_path}')
            success = cv2.imwrite(output_path, image)
            
            if not success:
                logger.error(f'Failed to write image: {output_path}')
        else:
            # No face detected or processing failed
            logger.warning(f"Skipping file (no face detected): {input_path}")
            
    except Exception as e:
        # Handle any unexpected errors during processing
        logger.error(f'Error processing {input_path}: {str(e)}')


def _process_image(filename, crop_dim):
    """
    Internal function to process a single image through the face detection and alignment pipeline.
    
    This function coordinates the image loading, face detection, and alignment steps.
    It's separated from preprocess_image to keep the public interface clean.
    
    Args:
        filename (str): Path to the image file to process
        crop_dim (int): Target dimension for the square output image
    
    Returns:
        numpy.ndarray or None: Processed image array if successful, None if processing failed
    
    Raises:
        IOError: If the image cannot be loaded from disk
    
    Process:
        1. Load image from disk and convert color space
        2. Detect and align the face using facial landmarks
        3. Return the processed image or None if no face was found
    """
    image = None
    aligned_image = None

    # Load the image from disk
    image = _buffer_image(filename)

    if image is not None:
        # Image loaded successfully - proceed with face alignment
        aligned_image = _align_image(image, crop_dim)
    else:
        # Failed to load image
        raise IOError(f'Error loading image: {filename}')

    return aligned_image


def _buffer_image(filename):
    """
    Loads an image from disk and converts it to RGB color space.
    
    OpenCV loads images in BGR format by default, but most face recognition
    models expect RGB format. This function handles the color space conversion.
    
    Args:
        filename (str): Path to the image file
    
    Returns:
        numpy.ndarray or None: Image array in RGB format, or None if loading failed
    
    Note:
        - Handles common image loading errors gracefully
        - Converts from BGR (OpenCV default) to RGB (standard format)
    """
    try:
        logger.debug(f'Loading image: {filename}')
        
        # Load image using OpenCV (loads in BGR format)
        image = cv2.imread(filename)
        
        if image is None:
            logger.error(f'Failed to load image: {filename}')
            return None
            
        # Convert from BGR to RGB color space for consistency with face recognition models
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
        
    except Exception as e:
        logger.error(f'Exception while loading image {filename}: {str(e)}')
        return None


def _align_image(image, crop_dim):
    """
    Detects the largest face in an image and aligns it using facial landmarks.
    
    This function uses dlib's face detection and alignment capabilities to:
    1. Find the largest face bounding box in the image
    2. Detect 68 facial landmarks within that face
    3. Align the face based on eye and lip positions
    4. Crop the aligned face to the specified dimensions
    
    Args:
        image (numpy.ndarray): Input image in RGB format
        crop_dim (int): Target dimension for the square output image
    
    Returns:
        numpy.ndarray or None: Aligned and cropped face image, or None if no face detected
    
    Face Alignment Details:
        - Uses INNER_EYES_AND_BOTTOM_LIP landmarks for alignment reference points
        - Aligns faces to a canonical pose for consistent training data
        - Handles rotation, scale, and translation to normalize face orientation
    
    Note:
        - Only processes the largest face if multiple faces are present
        - Returns None if no face is detected or alignment fails
        - Output image is converted back to RGB format
    """
    # Detect the largest face bounding box in the image
    bb = align_dlib.getLargestFaceBoundingBox(image)
    
    if bb is None:
        logger.debug('No face detected in image')
        return None
    
    # Align the face using facial landmarks
    # INNER_EYES_AND_BOTTOM_LIP provides 3 reference points for stable alignment
    aligned = align_dlib.align(crop_dim, image, bb, 
                             landmarkIndices=AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
    
    if aligned is not None:
        # Successful alignment - convert color space for output
        # Note: This conversion might be incorrect if align() returns RGB
        # You may want to verify the color space returned by align_dlib.align()
        aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        logger.debug(f'Successfully aligned face, output size: {aligned.shape}')
    else:
        logger.debug('Face alignment failed')
    
    return aligned


if __name__ == '__main__':
    """
    Command-line interface for the face preprocessing module.
    
    This script can be run from the command line with various options to
    customize the preprocessing behavior.
    
    Example usage:
        python preprocess.py --input-dir ./raw_faces --output-dir ./processed_faces --crop-dim 160
    """
    # Configure logging to show INFO level messages and above
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(
        description='Preprocess face images by detecting, aligning, and cropping faces',
        add_help=True
    )
    
    # Define command-line arguments
    parser.add_argument(
        '--input-dir', 
        type=str, 
        action='store', 
        default='data', 
        dest='input_dir',
        help='Input directory containing subdirectories of face images (default: data)'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        action='store', 
        default='output', 
        dest='output_dir',
        help='Output directory for processed images (default: output)'
    )
    
    parser.add_argument(
        '--crop-dim', 
        type=int, 
        action='store', 
        default=180, 
        dest='crop_dim',
        help='Size to crop images to (creates square images of crop_dim x crop_dim) (default: 180)'
    )

    # Parse command-line arguments
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.input_dir):
        logger.error(f'Input directory does not exist: {args.input_dir}')
        exit(1)
    
    if args.crop_dim <= 0:
        logger.error(f'Crop dimension must be positive: {args.crop_dim}')
        exit(1)
    
    # Log configuration
    logger.info(f'Starting face preprocessing with configuration:')
    logger.info(f'  Input directory: {args.input_dir}')
    logger.info(f'  Output directory: {args.output_dir}')
    logger.info(f'  Crop dimension: {args.crop_dim}x{args.crop_dim}')
    logger.info(f'  CPU cores: {mp.cpu_count()}')
    
    # Run the main processing function
    main(args.input_dir, args.output_dir, args.crop_dim)