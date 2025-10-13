"""
Data loader module for parsing image pair lists from CSV files and loading images.

This module provides functions to:
- Load and validate image pair data from CSV files
- Load images from disk with validation and format conversion
"""

import csv
import os
from pathlib import Path
from typing import List, Dict, Any, Union
import numpy as np
import cv2


def load_image_pairs_from_csv(csv_file_path: str) -> List[Dict[str, str]]:
    """
    Load image pair data from a CSV file.
    
    Args:
        csv_file_path: Path to the CSV file containing image pair data
        
    Returns:
        List of dictionaries, each containing:
        - image_pair_id: Unique identifier for the image pair
        - source_path: Path to the source image
        - target_path: Path to the target image
        
    Raises:
        FileNotFoundError: If the CSV file does not exist
        ValueError: If the CSV is malformed or missing required columns/values
    """
    # Check if file exists
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
    
    # Required columns
    required_columns = {"image_pair_id", "source_path", "target_path"}
    image_pairs = []
    
    try:
        with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
            # Use default CSV reader to avoid dialect detection issues
            reader = csv.DictReader(csvfile)
            
            # Check if all required columns are present
            if reader.fieldnames is None:
                raise ValueError("CSV file appears to be empty or has no header")
                
            csv_columns = set(reader.fieldnames)
            missing_columns = required_columns - csv_columns
            
            if missing_columns:
                missing_list = ", ".join(sorted(missing_columns))
                raise ValueError(f"CSV missing required column(s): {missing_list}")
            
            # Process each row
            for row_num, row in enumerate(reader, start=2):  # Start at 2 to account for header
                try:
                    # Check for malformed CSV (extra columns create None keys)
                    if None in row:
                        raise ValueError(f"Malformed CSV: extra columns detected at row {row_num}")
                    
                    # Check for empty values in required fields
                    for col in required_columns:
                        value = row.get(col, "").strip()
                        if not value:
                            raise ValueError(f"Empty value found in required column '{col}' at row {row_num}")
                    
                    # Create image pair dictionary
                    image_pair = {
                        "image_pair_id": row["image_pair_id"].strip(),
                        "source_path": row["source_path"].strip(),
                        "target_path": row["target_path"].strip()
                    }
                    
                    image_pairs.append(image_pair)
                    
                except csv.Error as e:
                    raise ValueError(f"Malformed CSV at row {row_num}: {e}")
                
    except csv.Error as e:
        raise ValueError(f"Invalid CSV format: {e}")
    except UnicodeDecodeError as e:
        raise ValueError(f"Unable to read CSV file (encoding issue): {e}")
    
    return image_pairs


def load_image(image_path: Union[str, Path], grayscale: bool = False) -> np.ndarray:
    """
    Load an image from disk with validation and format conversion.
    
    Args:
        image_path: Path to the image file (string or Path object)
        grayscale: If True, load image in grayscale mode; if False, load in color (RGB)
        
    Returns:
        Numpy ndarray containing the image data:
        - For color images: 3D array with shape (height, width, 3) and dtype uint8
        - For grayscale images: 2D array with shape (height, width) and dtype uint8
        
    Raises:
        FileNotFoundError: If the image file does not exist
        ValueError: If the image file is corrupted or cannot be read
        TypeError: If image_path is None or invalid type
    """
    # Validate input
    if image_path is None:
        raise TypeError("image_path cannot be None")
    
    # Convert to Path object for easier handling
    if isinstance(image_path, str):
        if not image_path:
            raise ValueError("image_path cannot be an empty string")
        path = Path(image_path)
    elif isinstance(image_path, Path):
        path = image_path
    else:
        raise TypeError(f"image_path must be a string or Path object, got {type(image_path)}")
    
    # Check if path exists
    if not path.exists():
        raise FileNotFoundError(f"Image file does not exist: {path}")
    
    # Check if path is a directory
    if path.is_dir():
        raise ValueError(f"Path is a directory, not a file: {path}")
    
    # Load image using OpenCV
    if grayscale:
        # Load in grayscale mode
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    else:
        # Load in color mode and convert from BGR to RGB
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Check if image was loaded successfully
    if image is None:
        raise ValueError(f"Failed to load image (file may be corrupted): {path}")
    
    return image