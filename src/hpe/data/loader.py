"""
Data loader module for parsing image pair lists from CSV files.

This module provides functions to load and validate image pair data from CSV files,
ensuring that all required fields are present and valid.
"""

import csv
import os
from typing import List, Dict, Any


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