"""
FaceNet Model Download and Extraction Utility

This script downloads pre-trained FaceNet models from Google Drive and extracts them
for use in face recognition tasks. The models are stored as compressed archives
and need to be extracted before use.

Original code adapted from:
https://github.com/davidsandberg/facenet/blob/master/src/download_and_extract_model.py

License: MIT
"""

import argparse
import logging
import os
import requests
import zipfile

# Dictionary mapping model names to their Google Drive file IDs
# Add new models here as they become available
MODEL_DICT = {
    '20170511-185253': '0B5MzpY9kBtDVOTVnU3NIaUdySFE'  # Pre-trained FaceNet model
}


def download_and_extract_model(model_name, data_dir):
    """
    Download and extract a FaceNet model from Google Drive.
    
    This function:
    1. Creates the destination directory if it doesn't exist
    2. Downloads the model zip file from Google Drive
    3. Extracts the contents to the specified directory
    
    Args:
        model_name (str): Name of the model to download (must be in MODEL_DICT)
        data_dir (str): Directory where the model will be extracted
        
    Raises:
        KeyError: If model_name is not found in MODEL_DICT
        
    Example:
        download_and_extract_model('20170511-185253', './models')
    """
    # Create data directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f'Created directory: {data_dir}')

    # Get Google Drive file ID for the specified model
    if model_name not in MODEL_DICT:
        raise KeyError(f'Model "{model_name}" not found. Available models: {list(MODEL_DICT.keys())}')
    
    file_id = MODEL_DICT[model_name]
    destination = os.path.join(data_dir, model_name + '.zip')
    
    # Download model if it doesn't already exist
    if not os.path.exists(destination):
        print(f'Downloading model "{model_name}" to {destination}')
        download_file_from_google_drive(file_id, destination)
        
        # Extract the downloaded zip file
        with zipfile.ZipFile(destination, 'r') as zip_ref:
            print(f'Extracting model to {data_dir}')
            zip_ref.extractall(data_dir)
            print(f'Successfully extracted {model_name}')
    else:
        print(f'Model {model_name} already exists at {destination}')


def download_file_from_google_drive(file_id, destination):
    """
    Download a file from Google Drive using its file ID.
    
    Handles Google Drive's virus scan warning for large files by
    detecting and using the confirmation token.
    
    Args:
        file_id (str): Google Drive file ID
        destination (str): Local path where file will be saved
        
    Note:
        Google Drive requires a confirmation token for large files
        to bypass the "virus scan" warning page.
    """
    # Google Drive download URL
    URL = "https://drive.google.com/uc?export=download"

    # Create a session to maintain cookies
    session = requests.Session()

    # Initial request to get potential confirmation token
    print("Initiating download from Google Drive...")
    response = session.get(URL, params={'id': file_id}, stream=True)
    
    # Check if we need a confirmation token (for large files)
    token = get_confirm_token(response)

    if token:
        print("Large file detected, using confirmation token...")
        # Request again with confirmation token
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    # Save the file content
    save_response_content(response, destination)
    print(f"Download completed: {destination}")


def get_confirm_token(response):
    """
    Extract confirmation token from Google Drive response cookies.
    
    Google Drive sets a cookie with a confirmation token when downloading
    large files to bypass the virus scan warning.
    
    Args:
        response (requests.Response): HTTP response from Google Drive
        
    Returns:
        str or None: Confirmation token if found, None otherwise
    """
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response, destination):
    """
    Save HTTP response content to a file in chunks.
    
    Downloads and saves the file in chunks to handle large files
    efficiently without loading the entire file into memory.
    
    Args:
        response (requests.Response): HTTP response containing file data
        destination (str): Path where the file will be saved
        
    Note:
        Uses 32KB chunks for efficient memory usage during download.
    """
    CHUNK_SIZE = 32768  # 32KB chunks for efficient memory usage

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # Filter out keep-alive chunks
                f.write(chunk)


def main():
    """
    Main function to handle command line arguments and initiate download.
    
    Command line usage:
        python download_and_extract_model.py --model-dir /path/to/models
    """
    # Set up logging for debug information
    logging.basicConfig(level=logging.DEBUG)
    
    # Set up command line argument parser
    parser = argparse.ArgumentParser(
        description='Download and extract FaceNet models from Google Drive',
        add_help=True
    )
    parser.add_argument(
        '--model-dir', 
        type=str, 
        action='store', 
        dest='model_dir',
        required=True,
        help='Directory path where the model will be downloaded and extracted'
    )

    # Parse command line arguments
    args = parser.parse_args()

    # Download and extract the default model
    # You can modify this to accept model name as an argument
    model_name = '20170511-185253'
    print(f'Starting download of model: {model_name}')
    download_and_extract_model(model_name, args.model_dir)
    print('Model download and extraction completed successfully!')


if __name__ == '__main__':
    main()