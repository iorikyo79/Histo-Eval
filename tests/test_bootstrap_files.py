"""
Bootstrap file existence tests for Histo-Eval project.

This test module verifies that all required project initialization files exist.
According to plan.md, this test should check for:
- requirements.txt
- config.yaml (template)
- .github/prd.md
- .github/tdd.md
- .github/instruction.md

Acceptance criteria: Clear assertion failure messages when files are missing.
"""

import os
import pytest


class TestBootstrapFiles:
    """Test that all required project initialization files exist."""
    
    def test_should_have_requirements_txt(self):
        """Requirements.txt file should exist in project root."""
        requirements_path = "requirements.txt"
        assert os.path.exists(requirements_path), f"Missing required file: {requirements_path}"
    
    def test_should_have_config_yaml_template(self):
        """Config.yaml template file should exist in project root."""
        config_path = "config.yaml"
        assert os.path.exists(config_path), f"Missing required file: {config_path}"
    
    def test_should_have_prd_document(self):
        """Product requirements document should exist in .github directory."""
        prd_path = ".github/prd.md"
        assert os.path.exists(prd_path), f"Missing required file: {prd_path}"
    
    def test_should_have_tdd_document(self):
        """TDD rules document should exist in .github directory."""
        tdd_path = ".github/tdd.md"
        assert os.path.exists(tdd_path), f"Missing required file: {tdd_path}"
    
    def test_should_have_instruction_document(self):
        """Instruction document should exist in .github directory."""
        instruction_path = ".github/copilot-instruction.md"
        assert os.path.exists(instruction_path), f"Missing required file: {instruction_path}"