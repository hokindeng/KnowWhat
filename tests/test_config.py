"""Basic tests for configuration."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def test_config_imports():
    """Test that config can be imported."""
    try:
        from config import (
            PROJECT_ROOT, DATA_DIR, HUMAN_RESULTS_DIR,
            EXPERIMENT_MAZES_DIR, MACHINE_RESULTS_DIR
        )
        assert True
    except ImportError as e:
        assert False, f"Failed to import config: {e}"

def test_paths_exist():
    """Test that critical paths exist."""
    from config import PROJECT_ROOT, DATA_DIR, CORE_DIR
    
    assert PROJECT_ROOT.exists(), f"Project root does not exist: {PROJECT_ROOT}"
    assert CORE_DIR.exists(), f"Core directory does not exist: {CORE_DIR}"
    # DATA_DIR might not exist yet, so we just check it's a Path object
    assert isinstance(DATA_DIR, Path), "DATA_DIR is not a Path object"

def test_logging_setup():
    """Test that logging can be set up."""
    from config import setup_logging
    
    logger = setup_logging("test_module")
    assert logger is not None
    logger.info("Test log message")

if __name__ == "__main__":
    test_config_imports()
    test_paths_exist()
    test_logging_setup()
    print("All tests passed!")
