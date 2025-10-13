"""
Data loader CSV parsing tests for Histo-Eval project.

This test module verifies CSV parsing functionality for image pair lists.
According to plan.md, this test should:
- Parse sample CSV to return [{image_pair_id, source_path, target_path}, ...] list
- Raise ValueError for missing required columns/empty values/wrong delimiter
- Show consistent exception messages for schema violations

Acceptance criteria: Consistent exception messages when schema is violated.
"""

import csv
import os
import pytest
from io import StringIO


class TestDataLoaderCSV:
    """Test CSV parsing functionality for image pair data."""
    
    def test_should_parse_valid_csv_to_image_pairs_list(self):
        """Should parse valid CSV file and return list of image pair dictionaries."""
        # Use fixture file
        fixture_path = "tests/fixtures/valid_pairs.csv"
        
        # Import the function to test
        from src.hpe.data.loader import load_image_pairs_from_csv
        
        # Parse CSV
        result = load_image_pairs_from_csv(fixture_path)
        
        # Verify result structure
        assert isinstance(result, list), "Result should be a list"
        assert len(result) == 3, "Should parse 3 image pairs"
        
        # Verify first pair structure
        first_pair = result[0]
        assert isinstance(first_pair, dict), "Each pair should be a dictionary"
        assert "image_pair_id" in first_pair, "Should have image_pair_id field"
        assert "source_path" in first_pair, "Should have source_path field" 
        assert "target_path" in first_pair, "Should have target_path field"
        
        # Verify first pair values
        assert first_pair["image_pair_id"] == "pair_001"
        assert first_pair["source_path"] == "/data/source1.png"
        assert first_pair["target_path"] == "/data/target1.png"
    
    def test_should_raise_error_for_missing_required_columns(self):
        """Should raise ValueError when required columns are missing."""
        # Use fixture file with missing target_path column
        fixture_path = "tests/fixtures/missing_column.csv"
        
        from src.hpe.data.loader import load_image_pairs_from_csv
        
        with pytest.raises(ValueError) as exc_info:
            load_image_pairs_from_csv(fixture_path)
        
        error_message = str(exc_info.value)
        assert "missing required column" in error_message.lower(), f"Error message should mention missing column: {error_message}"
        assert "target_path" in error_message, f"Error message should specify missing column name: {error_message}"
    
    def test_should_raise_error_for_empty_values(self):
        """Should raise ValueError when required fields have empty values."""
        # Use fixture file with empty values
        fixture_path = "tests/fixtures/empty_values.csv"
        
        from src.hpe.data.loader import load_image_pairs_from_csv
        
        with pytest.raises(ValueError) as exc_info:
            load_image_pairs_from_csv(fixture_path)
        
        error_message = str(exc_info.value)
        assert "empty value" in error_message.lower(), f"Error message should mention empty value: {error_message}"
    
    def test_should_raise_error_for_malformed_csv(self):
        """Should raise ValueError for malformed CSV files."""
        # Use fixture file with extra columns
        fixture_path = "tests/fixtures/malformed.csv"
        
        from src.hpe.data.loader import load_image_pairs_from_csv
        
        with pytest.raises(ValueError) as exc_info:
            load_image_pairs_from_csv(fixture_path)
        
        error_message = str(exc_info.value)
        assert "malformed" in error_message.lower() or "extra" in error_message.lower(), \
            f"Error message should indicate malformed CSV: {error_message}"
    
    def test_should_handle_file_not_found(self):
        """Should raise FileNotFoundError for non-existent files."""
        from src.hpe.data.loader import load_image_pairs_from_csv
        
        with pytest.raises(FileNotFoundError):
            load_image_pairs_from_csv("/non/existent/file.csv")