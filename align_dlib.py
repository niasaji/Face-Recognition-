"""
Face Alignment Module using dlib

This module provides face alignment functionality using dlib's landmark detection.
It's designed to preprocess faces for neural network input by normalizing their
position, size, and orientation.

Original code adapted from OpenFace project:
https://github.com/cmusatyalab/openface/blob/master/openface/align_dlib.py

License: Apache 2.0
"""

import cv2
import dlib
import numpy as np

# Normalized facial landmark template - defines the standard positions
# for 68 facial landmarks in a normalized coordinate system (0-1 range)
# These coordinates represent the "ideal" face shape that all faces will be aligned to
TEMPLATE = np.float32([
    (0.0792396913815, 0.339223741112), (0.0829219487236, 0.456955367943),
    (0.0967927109165, 0.575648016728), (0.122141515615, 0.691921601066),
    (0.168687863544, 0.800341263616), (0.239789390707, 0.895732504778),
    (0.325662452515, 0.977068762493), (0.422318282013, 1.04329000149),
    (0.531777802068, 1.06080371126), (0.641296298053, 1.03981924107),
    (0.738105872266, 0.972268833998), (0.824444363295, 0.889624082279),
    (0.894792677532, 0.792494155836), (0.939395486253, 0.681546643421),
    (0.96111933829, 0.562238253072), (0.970579841181, 0.441758925744),
    (0.971193274221, 0.322118743967), (0.163846223133, 0.249151738053),
    (0.21780354657, 0.204255863861), (0.291299351124, 0.192367318323),
    (0.367460241458, 0.203582210627), (0.4392945113, 0.233135599851),
    (0.586445962425, 0.228141644834), (0.660152671635, 0.195923841854),
    (0.737466449096, 0.182360984545), (0.813236546239, 0.192828009114),
    (0.8707571886, 0.235293377042), (0.51534533827, 0.31863546193),
    (0.516221448289, 0.396200446263), (0.517118861835, 0.473797687758),
    (0.51816430343, 0.553157797772), (0.433701156035, 0.604054457668),
    (0.475501237769, 0.62076344024), (0.520712933176, 0.634268222208),
    (0.565874114041, 0.618796581487), (0.607054002672, 0.60157671656),
    (0.252418718401, 0.331052263829), (0.298663015648, 0.302646354002),
    (0.355749724218, 0.303020650651), (0.403718978315, 0.33867711083),
    (0.352507175597, 0.349987615384), (0.296791759886, 0.350478978225),
    (0.631326076346, 0.334136672344), (0.679073381078, 0.29645404267),
    (0.73597236153, 0.294721285802), (0.782865376271, 0.321305281656),
    (0.740312274764, 0.341849376713), (0.68499850091, 0.343734332172),
    (0.353167761422, 0.746189164237), (0.414587777921, 0.719053835073),
    (0.477677654595, 0.706835892494), (0.522732900812, 0.717092275768),
    (0.569832064287, 0.705414478982), (0.635195811927, 0.71565572516),
    (0.69951672331, 0.739419187253), (0.639447159575, 0.805236879972),
    (0.576410514055, 0.835436670169), (0.525398405766, 0.841706377792),
    (0.47641545769, 0.837505914975), (0.41379548902, 0.810045601727),
    (0.380084785646, 0.749979603086), (0.477955996282, 0.74513234612),
    (0.523389793327, 0.748924302636), (0.571057789237, 0.74332894691),
    (0.672409137852, 0.744177032192), (0.572539621444, 0.776609286626),
    (0.5240106503, 0.783370783245), (0.477561227414, 0.778476346951)])

# Inverse transformation matrix for coordinate conversion
# Used for converting between different coordinate systems
INV_TEMPLATE = np.float32([
    (-0.04099179660567834, -0.008425234314031194, 2.575498465013183),
    (0.04062510634554352, -0.009678089746831375, -1.2534351452524177),
    (0.0003666902601348179, 0.01810332406086298, -0.32206331976076663)])

# Calculate min/max values for template normalization
# This creates a normalized template where coordinates are scaled to [0,1] range
TPL_MIN, TPL_MAX = np.min(TEMPLATE, axis=0), np.max(TEMPLATE, axis=0)
MINMAX_TEMPLATE = (TEMPLATE - TPL_MIN) / (TPL_MAX - TPL_MIN)


class AlignDlib:
    """
    Face alignment class using dlib's facial landmark detection.
    
    This class provides functionality to detect faces, find facial landmarks,
    and align faces to a standard pose for consistent neural network input.
    
    The alignment process:
    1. Detect faces in the image
    2. Find 68 facial landmarks using dlib's predictor
    3. Transform the face to match a standard template
    4. Crop and resize to specified dimensions
    
    Attributes:
        INNER_EYES_AND_BOTTOM_LIP: Landmark indices for basic alignment (3 points)
        OUTER_EYES_AND_NOSE: Alternative landmark indices for alignment (3 points)
    """

    # Predefined landmark index sets for different alignment strategies
    INNER_EYES_AND_BOTTOM_LIP = [39, 42, 57]  # More stable alignment points
    OUTER_EYES_AND_NOSE = [36, 45, 33]        # Alternative alignment points

    def __init__(self, facePredictor):
        """
        Initialize the AlignDlib object with a dlib face predictor.

        Args:
            facePredictor (str): Path to dlib's shape predictor model file
                                (e.g., 'shape_predictor_68_face_landmarks.dat')
        
        Raises:
            AssertionError: If facePredictor is None
        """
        assert facePredictor is not None, "Face predictor path cannot be None"

        # Initialize dlib's face detector (HOG-based)
        self.detector = dlib.get_frontal_face_detector()
        
        # Initialize dlib's landmark predictor with the provided model
        self.predictor = dlib.shape_predictor(facePredictor)

    def getAllFaceBoundingBoxes(self, rgbImg):
        """
        Detect all faces in an RGB image.

        Args:
            rgbImg (numpy.ndarray): RGB image array with shape (height, width, 3)

        Returns:
            dlib.rectangles: List of detected face bounding boxes
                           Empty list if no faces found or if an error occurs

        Note:
            Uses dlib's get_frontal_face_detector() with upsampling factor of 1
        """
        assert rgbImg is not None, "Input image cannot be None"

        try:
            # Run face detection with upsampling factor of 1
            # Higher values increase detection accuracy but slow down processing
            return self.detector(rgbImg, 1)
        except Exception as e:
            print(f"Warning: Error during face detection: {e}")
            # Return empty list on any exception to prevent crashes
            return []

    def getLargestFaceBoundingBox(self, rgbImg, skipMulti=False):
        """
        Find the largest face bounding box in an image.

        Args:
            rgbImg (numpy.ndarray): RGB image array with shape (height, width, 3)
            skipMulti (bool): If True, return None when multiple faces are detected
                            If False, return largest face even with multiple detections

        Returns:
            dlib.rectangle or None: Largest face bounding box, or None if:
                                  - No faces detected
                                  - Multiple faces detected and skipMulti=True
        """
        assert rgbImg is not None, "Input image cannot be None"

        # Get all face detections
        faces = self.getAllFaceBoundingBoxes(rgbImg)
        
        # Determine if we should return a face based on detection count and skipMulti flag
        if (not skipMulti and len(faces) > 0) or len(faces) == 1:
            # Return the face with maximum area (width * height)
            return max(faces, key=lambda rect: rect.width() * rect.height())
        else:
            return None

    def findLandmarks(self, rgbImg, bb):
        """
        Find facial landmarks within a given bounding box.

        Args:
            rgbImg (numpy.ndarray): RGB image array with shape (height, width, 3)
            bb (dlib.rectangle): Face bounding box from face detection

        Returns:
            list: List of (x, y) tuples representing the 68 facial landmark coordinates

        Note:
            Returns 68 landmark points following dlib's facial landmark model:
            - Points 0-16: Jaw line
            - Points 17-21: Right eyebrow
            - Points 22-26: Left eyebrow
            - Points 27-35: Nose
            - Points 36-41: Right eye
            - Points 42-47: Left eye
            - Points 48-67: Mouth
        """
        assert rgbImg is not None, "Input image cannot be None"
        assert bb is not None, "Bounding box cannot be None"

        # Use dlib's shape predictor to find landmarks within the bounding box
        points = self.predictor(rgbImg, bb)
        
        # Convert dlib points to list of (x, y) tuples
        return [(p.x, p.y) for p in points.parts()]

    def align(self, imgDim, rgbImg, bb=None, landmarks=None, 
              landmarkIndices=None, skipMulti=False, scale=1.0):
        """
        Align and crop a face from an image to standard dimensions.

        This is the main alignment function that:
        1. Detects face (if bb not provided)
        2. Finds landmarks (if landmarks not provided)  
        3. Computes affine transformation to standard pose
        4. Applies transformation and crops to specified size

        Args:
            imgDim (int): Output image dimensions (creates imgDim x imgDim square)
            rgbImg (numpy.ndarray): Input RGB image with shape (height, width, 3)
            bb (dlib.rectangle, optional): Face bounding box. Auto-detected if None
            landmarks (list, optional): Facial landmarks as (x,y) tuples. Auto-detected if None
            landmarkIndices (list, optional): Indices of landmarks to use for alignment.
                                            Defaults to INNER_EYES_AND_BOTTOM_LIP
            skipMulti (bool): Skip processing if multiple faces detected
            scale (float): Scale factor for the aligned face (1.0 = no scaling)

        Returns:
            numpy.ndarray or None: Aligned RGB face image with shape (imgDim, imgDim, 3)
                                 Returns None if alignment fails

        Example:
            aligned_face = aligner.align(160, rgb_image)  # 160x160 aligned face
        """
        assert imgDim is not None, "Image dimension cannot be None"
        assert rgbImg is not None, "Input image cannot be None"
        
        # Use default landmark indices if not specified
        if landmarkIndices is None:
            landmarkIndices = self.INNER_EYES_AND_BOTTOM_LIP

        assert landmarkIndices is not None, "Landmark indices cannot be None"

        # Detect face if bounding box not provided
        if bb is None:
            bb = self.getLargestFaceBoundingBox(rgbImg, skipMulti)
            if bb is None:
                return None  # No face detected or multiple faces with skipMulti=True

        # Find landmarks if not provided
        if landmarks is None:
            landmarks = self.findLandmarks(rgbImg, bb)

        # Convert landmarks to numpy array for computation
        npLandmarks = np.float32(landmarks)
        npLandmarkIndices = np.array(landmarkIndices)

        # Compute affine transformation matrix
        # Maps selected landmarks to their corresponding positions in the standard template
        source_points = npLandmarks[npLandmarkIndices]  # Current landmark positions
        target_points = (imgDim * MINMAX_TEMPLATE[npLandmarkIndices] * scale + 
                        imgDim * (1 - scale) / 2)  # Target positions in output image
        
        # Calculate 2x3 affine transformation matrix
        H = cv2.getAffineTransform(source_points, target_points)
        
        # Apply transformation and crop to specified dimensions
        thumbnail = cv2.warpAffine(rgbImg, H, (imgDim, imgDim))

        return thumbnail