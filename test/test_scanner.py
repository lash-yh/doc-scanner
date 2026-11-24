import importlib
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

# The current source mixes package and standalone imports. Allow behavioral
# tests to run, and separately verify normal imports without this workaround.
with patch.object(sys, "path", [str(ROOT / "src"), str(ROOT / "src/core"), *sys.path]):
    scanner = importlib.import_module("src.core.scanner").scanner
    Intersection = importlib.import_module("entities").Intersection


class ImportTests(unittest.TestCase):
    def test_modules_import_from_repository_root(self):
        for module in ("src.core.entities", "src.core.scanner"):
            with self.subTest(module=module):
                result = subprocess.run(
                    [sys.executable, "-I", "-c", f"import sys; sys.path.insert(0, {str(ROOT)!r}); import {module}"],
                    cwd=ROOT, capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr)


class IntersectionTests(unittest.TestCase):
    def test_accepts_explicit_image_width(self):
        result = Intersection((2, 3), (0, 2), (np.pi / 2, 3), x=100)
        self.assertEqual(result.intersection, (2, 3))
        self.assertEqual(result.x, 100)
        self.assertIsNone(result.image)

    def test_rejects_invalid_coordinate_and_lines(self):
        for point, vertical, horizontal in [((1,), (0, 2), (1, 3)), ((1, 2), (0,), (1, 3)), ((1, 2), (0, 2), (1,))]:
            with self.subTest(point=point, vertical=vertical, horizontal=horizontal), self.assertRaises(ValueError):
                Intersection(point, vertical, horizontal, x=100)

    def test_requires_image_or_width(self):
        with self.assertRaises(ValueError):
            Intersection((2, 3), (0, 2), (np.pi / 2, 3))

    def test_accepts_numpy_image(self):
        image = np.zeros((60, 100), dtype=np.uint8)
        result = Intersection((2, 3), (0, 2), (np.pi / 2, 3), image=image)
        self.assertIs(result.image, image)
        self.assertEqual(result.x, 100)


class ScannerTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((160, 200), dtype=np.uint8)
        cv2.rectangle(self.image, (30, 30), (170, 130), 255, -1)
        self.subject = scanner(self.image)

    def test_constructor_accepts_grayscale(self):
        self.assertIs(self.subject.image, self.image)

    def test_constructor_rejects_color_and_one_dimensional_images(self):
        for image in (np.zeros((20, 20, 3)), np.zeros(20)):
            with self.subTest(shape=image.shape), self.assertRaises(ValueError):
                scanner(image)

    def test_preprocess_produces_edges_without_modifying_input(self):
        original = self.image.copy()
        self.subject.preprocess()
        np.testing.assert_array_equal(self.image, original)
        for name in ("blurred", "hist_equalized", "filterred", "edges_img", "edges_img_dilated"):
            with self.subTest(name=name):
                result = getattr(self.subject, name)
                self.assertEqual(result.shape, self.image.shape)
                self.assertTrue(np.isfinite(result).all())
        self.assertGreater(np.count_nonzero(self.subject.edges_img), 0)
        self.assertGreaterEqual(np.count_nonzero(self.subject.edges_img_dilated), np.count_nonzero(self.subject.edges_img))

    def test_intensity_mask_excludes_pixels(self):
        self.subject.preprocess(intensity_lower=1, intensity_upper=254)
        self.assertEqual(np.count_nonzero(self.subject.filterred), 0)

    def test_scan_runs_preprocessing(self):
        self.subject.scan()
        self.assertEqual(self.subject.edges_img.shape, self.image.shape)

    def test_hough_detects_rectangle_line_families(self):
        self.subject.preprocess()
        lines = self.subject.hough_transform()
        self.assertEqual(set(lines), {"v", "h", "o"})
        self.assertGreaterEqual(len(lines["v"]), 2)
        self.assertGreaterEqual(len(lines["h"]), 2)

    def test_blank_image_has_no_lines(self):
        subject = scanner(np.zeros_like(self.image))
        subject.preprocess()
        self.assertEqual(subject.hough_transform(), {"v": [], "h": [], "o": []})

    def test_no_line_pairs_returns_empty_intersections(self):
        self.subject.lines = {"v": [], "h": [], "o": []}
        self.assertEqual(self.subject.clac_intersections(), [])

    def test_intersections_from_known_lines(self):
        # Use nonzero angles so conversion to Cartesian points is finite.
        self.subject.edges_img = self.image
        self.subject.lines = {"v": [(np.pi / 6, 10)], "h": [(np.pi / 2, 4)], "o": []}
        intersections = self.subject.clac_intersections()
        self.assertEqual(len(intersections), 1)
        np.testing.assert_allclose(intersections[0].intersection, (16 / np.sqrt(3), 4))

    def test_connectivity_with_no_intersections(self):
        self.subject.intersections = []
        self.assertEqual(self.subject.calc_connectivity(), [])

    def test_connectivity_for_existing_intersection(self):
        self.subject.edges_img_dilated = self.image
        self.subject.intersections = [Intersection((30, 30), (0, 30), (np.pi / 2, 30), x=200)]
        result = self.subject.calc_connectivity()
        self.assertEqual(len(result), 1)
        self.assertIs(result[0].image, self.image)
