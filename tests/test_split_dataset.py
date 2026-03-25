"""
Tests for train/val dataset split script.
Critical: no data leakage between train and val sets.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from split_dataset import compute_split, execute_split


class TestComputeSplit:
    """Tests for computing which files go to train vs val."""

    def test_80_20_split_ratio(self):
        """Should split approximately 80/20."""
        stems = [f"img_{i:03d}" for i in range(100)]
        train, val = compute_split(stems, val_ratio=0.2, seed=42)
        assert len(train) == 80
        assert len(val) == 20

    def test_no_overlap(self):
        """Train and val sets must have zero overlap."""
        stems = [f"img_{i:03d}" for i in range(50)]
        train, val = compute_split(stems, val_ratio=0.2, seed=42)
        assert set(train).isdisjoint(set(val))

    def test_complete_coverage(self):
        """Union of train and val must equal the full input."""
        stems = [f"img_{i:03d}" for i in range(50)]
        train, val = compute_split(stems, val_ratio=0.2, seed=42)
        assert set(train) | set(val) == set(stems)

    def test_deterministic_with_seed(self):
        """Same seed should produce same split."""
        stems = [f"img_{i:03d}" for i in range(50)]
        train1, val1 = compute_split(stems, val_ratio=0.2, seed=42)
        train2, val2 = compute_split(stems, val_ratio=0.2, seed=42)
        assert train1 == train2
        assert val1 == val2

    def test_different_seed_different_split(self):
        """Different seeds should (almost certainly) produce different splits."""
        stems = [f"img_{i:03d}" for i in range(50)]
        train1, _ = compute_split(stems, val_ratio=0.2, seed=42)
        train2, _ = compute_split(stems, val_ratio=0.2, seed=99)
        assert train1 != train2

    def test_small_dataset(self):
        """With 5 items at 20%, should get 4 train + 1 val."""
        stems = ["a", "b", "c", "d", "e"]
        train, val = compute_split(stems, val_ratio=0.2, seed=42)
        assert len(train) == 4
        assert len(val) == 1

    def test_single_item(self):
        """Single item should go to train (val would be empty)."""
        stems = ["only_one"]
        train, val = compute_split(stems, val_ratio=0.2, seed=42)
        assert len(train) == 1
        assert len(val) == 0

    def test_empty_input(self):
        """Empty input should return empty lists."""
        train, val = compute_split([], val_ratio=0.2, seed=42)
        assert train == []
        assert val == []


class TestExecuteSplit:
    """Tests for actually copying files into train/val directories."""

    def _setup_files(self, tmp_path, count=10):
        """Create dummy image and label files."""
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()

        stems = []
        for i in range(count):
            stem = f"rs_{i:03d}"
            (images_dir / f"{stem}.png").write_text(f"image_{i}")
            (labels_dir / f"{stem}.txt").write_text(f"0 0.1 0.2 0.3 0.2 0.3 0.4 0.1 0.4")
            stems.append(stem)

        return images_dir, labels_dir, stems

    def test_files_copied_to_correct_dirs(self, tmp_path):
        """Train images go to train/, val images go to val/."""
        images_dir, labels_dir, stems = self._setup_files(tmp_path, count=10)
        output_dir = tmp_path / "dataset"

        train_stems, val_stems = compute_split(stems, val_ratio=0.2, seed=42)
        execute_split(
            train_stems=train_stems,
            val_stems=val_stems,
            images_src=images_dir,
            labels_src=labels_dir,
            output_base=output_dir,
            img_ext=".png",
            label_ext=".txt",
        )

        for stem in train_stems:
            assert (output_dir / "images" / "train" / f"{stem}.png").exists()
            assert (output_dir / "labels" / "train" / f"{stem}.txt").exists()

        for stem in val_stems:
            assert (output_dir / "images" / "val" / f"{stem}.png").exists()
            assert (output_dir / "labels" / "val" / f"{stem}.txt").exists()

    def test_no_files_in_wrong_split(self, tmp_path):
        """Train files must NOT appear in val dir and vice versa."""
        images_dir, labels_dir, stems = self._setup_files(tmp_path, count=10)
        output_dir = tmp_path / "dataset"

        train_stems, val_stems = compute_split(stems, val_ratio=0.2, seed=42)
        execute_split(
            train_stems=train_stems,
            val_stems=val_stems,
            images_src=images_dir,
            labels_src=labels_dir,
            output_base=output_dir,
            img_ext=".png",
            label_ext=".txt",
        )

        train_img_names = {f.name for f in (output_dir / "images" / "train").iterdir()}
        val_img_names = {f.name for f in (output_dir / "images" / "val").iterdir()}
        assert train_img_names.isdisjoint(val_img_names)

    def test_file_content_preserved(self, tmp_path):
        """Copied files should have identical content to originals."""
        images_dir, labels_dir, stems = self._setup_files(tmp_path, count=5)
        output_dir = tmp_path / "dataset"

        train_stems, val_stems = compute_split(stems, val_ratio=0.2, seed=42)
        execute_split(
            train_stems=train_stems,
            val_stems=val_stems,
            images_src=images_dir,
            labels_src=labels_dir,
            output_base=output_dir,
            img_ext=".png",
            label_ext=".txt",
        )

        for stem in train_stems + val_stems:
            split = "train" if stem in train_stems else "val"
            original = (images_dir / f"{stem}.png").read_text()
            copied = (output_dir / "images" / split / f"{stem}.png").read_text()
            assert original == copied
