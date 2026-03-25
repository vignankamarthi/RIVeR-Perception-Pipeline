"""
Tests for capture_utils.py -- shared logic for capture scripts.
Pure logic tests. No hardware dependencies.
"""

import sys
from pathlib import Path

import pytest

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from capture_utils import get_next_index, make_filename


class TestGetNextIndex:
    """Tests for get_next_index -- finds next available filename index."""

    def test_empty_directory(self, tmp_path):
        """Empty dir should return index 1."""
        assert get_next_index(tmp_path, "rs") == 1

    def test_sequential_files(self, tmp_path):
        """With rs_001.png through rs_005.png, should return 6."""
        for i in range(1, 6):
            (tmp_path / f"rs_{i:03d}.png").touch()
        assert get_next_index(tmp_path, "rs") == 6

    def test_gap_in_sequence(self, tmp_path):
        """With rs_001.png and rs_003.png (gap at 2), should return 4 (max+1)."""
        (tmp_path / "rs_001.png").touch()
        (tmp_path / "rs_003.png").touch()
        assert get_next_index(tmp_path, "rs") == 4

    def test_ignores_non_matching_prefix(self, tmp_path):
        """Should ignore files with different prefix."""
        (tmp_path / "rs_001.png").touch()
        (tmp_path / "kt_002.png").touch()
        assert get_next_index(tmp_path, "rs") == 2

    def test_ignores_non_png_files(self, tmp_path):
        """Should ignore non-PNG files even with matching prefix."""
        (tmp_path / "rs_001.png").touch()
        (tmp_path / "rs_002.txt").touch()
        assert get_next_index(tmp_path, "rs") == 2

    def test_single_file(self, tmp_path):
        """With only rs_001.png, should return 2."""
        (tmp_path / "rs_001.png").touch()
        assert get_next_index(tmp_path, "rs") == 2

    def test_kinect_prefix(self, tmp_path):
        """Works with kt prefix for Kinect files."""
        (tmp_path / "kt_001.png").touch()
        (tmp_path / "kt_002.png").touch()
        assert get_next_index(tmp_path, "kt") == 3

    def test_mixed_prefixes_isolated(self, tmp_path):
        """Each prefix tracks independently."""
        (tmp_path / "rs_001.png").touch()
        (tmp_path / "rs_002.png").touch()
        (tmp_path / "kt_001.png").touch()
        (tmp_path / "kt_005.png").touch()
        assert get_next_index(tmp_path, "rs") == 3
        assert get_next_index(tmp_path, "kt") == 6

    def test_malformed_filename_ignored(self, tmp_path):
        """Files like rs_abc.png should be ignored gracefully."""
        (tmp_path / "rs_abc.png").touch()
        (tmp_path / "rs_001.png").touch()
        assert get_next_index(tmp_path, "rs") == 2


class TestMakeFilename:
    """Tests for make_filename -- generates zero-padded filenames."""

    def test_basic_format(self):
        """Should produce rs_001.png format."""
        assert make_filename("rs", 1) == "rs_001.png"

    def test_zero_padded(self):
        """Single-digit index should be zero-padded to 3 digits."""
        assert make_filename("rs", 5) == "rs_005.png"

    def test_double_digit(self):
        """Double-digit index should be zero-padded to 3 digits."""
        assert make_filename("rs", 42) == "rs_042.png"

    def test_triple_digit(self):
        """Triple-digit index needs no padding."""
        assert make_filename("rs", 100) == "rs_100.png"

    def test_kinect_prefix(self):
        """Works with kt prefix."""
        assert make_filename("kt", 7) == "kt_007.png"

    def test_high_index(self):
        """Four-digit index should still work (no truncation)."""
        assert make_filename("rs", 1234) == "rs_1234.png"
