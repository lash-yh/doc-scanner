"""Scanner behavior and integration diagnostics, without changing production.

Load complete private module instances to avoid the known mixed imports.
Frame, Intersection and all scanner methods execute their current code.
These private loads do not replace canonical modules or alter other tests.
"""

from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch
import warnings

import cv2
import numpy as np

from src import core
from src.core import math_utils


ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def temporary_module_aliases(aliases):
    # Restore only aliases, keeping newly loaded native dependencies in place.
    # Restoring the entire sys.modules inventory can unload NumPy extensions
    # that cannot be initialized a second time in the same process.
    missing = object()
    previous = {key: sys.modules.get(key, missing) for key in aliases}
    sys.modules.update(aliases)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is missing:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


def load_private_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


with temporary_module_aliases({"math_utils": math_utils}):
    entities = load_private_module("_scanner_diagnostic_entities", "src/core/entities.py")

with temporary_module_aliases({
    "core": core, "core.math_utils": math_utils,
    "core.entities": entities,
    "math_utils": math_utils, "entities": entities,
}):
    scanner_module = load_private_module("_scanner_diagnostic_module", "src/core/scanner.py")

Scanner = scanner_module.scanner


def rectangle_image():
    image = np.zeros((160, 200), dtype=np.uint8)
    cv2.rectangle(image, (30, 30), (170, 130), 255, -1)
    return image


def subject_with_lines():
    subject = Scanner(np.zeros((40, 60), dtype=np.uint8))
    subject.edges_img = np.ones((40, 60), dtype=np.uint8)
    subject.edges_img_dilated = np.ones((40, 60), dtype=np.uint8)
    subject.lines = {"v": [(np.pi / 6, 20)], "h": [(np.pi / 2, 10)], "o": []}
    return subject


class ConstructorDiagnostics(unittest.TestCase):
    def test_grayscale_image_is_stored_by_reference(self):
        image = rectangle_image()
        self.assertIs(Scanner(image).image, image)

    def test_rejects_non_two_dimensional_arrays(self):
        for image in (np.zeros(3), np.zeros((3, 3, 3)), np.array(1)):
            with self.subTest(shape=image.shape), self.assertRaises(ValueError):
                Scanner(image)

    def test_non_arrays_need_explicit_validation(self):
        for image in (None, [[0, 1], [1, 0]]):
            with self.subTest(image=image), self.assertRaises(ValueError):
                Scanner(image)

    def test_empty_images_need_explicit_validation(self):
        for shape in ((0, 20), (20, 0)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                Scanner(np.zeros(shape, dtype=np.uint8))

    def test_non_uint8_inputs_are_accepted_but_fail_preprocessing(self):
        # Characterize the current contract gap, not a requirement to support
        # every dtype. The public supported types still need to be defined.
        for dtype in (np.uint16, np.float32, np.float64, np.bool_):
            with self.subTest(dtype=dtype):
                subject = Scanner(np.zeros((40, 40), dtype=dtype))
                with self.assertRaises(cv2.error):
                    subject.preprocess()


class ScanDiagnostics(unittest.TestCase):
    def test_scan_currently_only_runs_preprocess_and_returns_none(self):
        subject = Scanner(rectangle_image())
        with patch.object(subject, "preprocess") as preprocess, \
             patch.object(subject, "hough_transform") as hough, \
             patch.object(subject, "clac_intersections") as intersections, \
             patch.object(subject, "calc_connectivity") as connectivity:
            self.assertIsNone(subject.scan())
            preprocess.assert_called_once_with()
            hough.assert_not_called()
            intersections.assert_not_called()
            connectivity.assert_not_called()


class PreprocessDiagnostics(unittest.TestCase):
    def test_regular_pipeline_preserves_input_and_shape(self):
        image = rectangle_image()
        original = image.copy()
        subject = Scanner(image)
        self.assertIsNone(subject.preprocess())
        np.testing.assert_array_equal(image, original)
        for name in ("blurred", "hist_equalized", "filterred", "edges_img", "edges_img_dilated"):
            with self.subTest(name=name):
                result = getattr(subject, name)
                self.assertEqual(result.shape, image.shape)
                self.assertTrue(np.isfinite(result).all())
        self.assertGreater(np.count_nonzero(subject.edges_img), 0)

    def test_intensity_mask_can_exclude_all_pixels(self):
        subject = Scanner(rectangle_image())
        subject.preprocess(intensity_lower=1, intensity_upper=254)
        self.assertEqual(np.count_nonzero(subject.filterred), 0)

    def test_blank_image_has_no_edges(self):
        subject = Scanner(np.zeros((40, 60), dtype=np.uint8))
        subject.preprocess()
        self.assertEqual(np.count_nonzero(subject.edges_img), 0)
        self.assertEqual(np.count_nonzero(subject.edges_img_dilated), 0)

    def test_kernel_size_changes_morphology_but_not_median_blur(self):
        with patch.object(scanner_module.cv2, "medianBlur", wraps=cv2.medianBlur) as blur:
            Scanner(rectangle_image()).preprocess(kernel_size=3)
            self.assertEqual(blur.call_args.args[1], 25)

    def test_current_edge_storage_is_float64_with_intensity_128(self):
        subject = Scanner(rectangle_image())
        subject.preprocess()
        self.assertEqual(subject.edges_img.dtype, np.float64)
        self.assertEqual(set(np.unique(subject.edges_img)), {0, 128})

    def test_dilation_is_computed_before_erosion(self):
        subject = Scanner(rectangle_image())
        subject.preprocess(erode_ks=1)
        original_dilated = subject.edges_img_dilated.copy()
        original_edges = subject.edges_img.copy()
        subject.preprocess(erode_ks=5)
        np.testing.assert_array_equal(subject.edges_img_dilated, original_dilated)
        self.assertLess(np.count_nonzero(subject.edges_img), np.count_nonzero(original_edges))

    def test_preprocessing_again_should_invalidate_downstream_results(self):
        subject = Scanner(rectangle_image())
        subject.preprocess()
        subject.hough_transform()
        subject.intersections = [object()]
        subject.image.fill(0)
        subject.preprocess()
        self.assertEqual(np.count_nonzero(subject.edges_img), 0)
        for name in ("lines", "intersections"):
            with self.subTest(stale_attribute=name):
                self.assertFalse(getattr(subject, name, None))


class HoughDiagnostics(unittest.TestCase):
    def test_real_rectangle_detects_both_line_families(self):
        subject = Scanner(rectangle_image())
        subject.preprocess()
        lines = subject.hough_transform()
        self.assertEqual(set(lines), {"v", "h", "o"})
        self.assertGreaterEqual(len(lines["v"]), 2)
        self.assertGreaterEqual(len(lines["h"]), 2)

    def test_blank_image_returns_three_empty_families(self):
        subject = Scanner(np.zeros((40, 60), dtype=np.uint8))
        subject.preprocess()
        self.assertEqual(subject.hough_transform(), {"v": [], "h": [], "o": []})

    def test_detected_angles_are_classified_without_swapping_rho(self):
        subject = subject_with_lines()
        angles = np.array([0, np.pi / 2, np.pi / 4, -np.pi / 12])
        rhos = np.array([10, 20, 30, 40])
        with patch.object(scanner_module, "hough_line_peaks",
                          return_value=(np.ones(4), angles, rhos)):
            lines = subject.hough_transform()
        self.assertEqual(lines["v"], [(0, 10)])
        self.assertEqual(lines["h"], [(np.pi / 2, 20)])
        self.assertEqual(lines["o"], [(np.pi / 4, 30), (-np.pi / 12, 40)])

    def test_ks_currently_has_no_effect_on_peak_arguments(self):
        subject = subject_with_lines()
        calls = []
        with patch.object(scanner_module, "hough_line_peaks", return_value=([], [], [])) as peaks:
            for ks in (1, 9):
                subject.hough_transform(ks=ks)
                calls.append(peaks.call_args.kwargs)
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(calls[0]["min_distance"], 10)
        self.assertEqual(calls[0]["min_angle"], 50)

    def test_threshold_is_relative_to_accumulator_maximum(self):
        subject = subject_with_lines()
        with patch.object(scanner_module, "hough_line", return_value=(np.array([[4, 10]]), [], [])), \
             patch.object(scanner_module, "hough_line_peaks", return_value=([], [], [])) as peaks:
            subject.hough_transform(threshold=0.3)
        self.assertAlmostEqual(peaks.call_args.kwargs["threshold"], 3)

    def test_negative_threshold_needs_explicit_validation(self):
        with self.assertRaises(ValueError):
            subject_with_lines().hough_transform(threshold=-1)

    def test_requires_preprocessed_edges(self):
        with self.assertRaises(ValueError):
            Scanner(rectangle_image()).hough_transform()

    def test_new_hough_results_should_invalidate_old_intersections(self):
        subject = subject_with_lines()
        subject.intersections = [object()]
        subject.edges_img.fill(0)
        self.assertEqual(subject.hough_transform(), {"v": [], "h": [], "o": []})
        self.assertFalse(getattr(subject, "intersections", None))


class IntersectionDiagnostics(unittest.TestCase):
    def test_empty_or_missing_line_family_returns_empty_list(self):
        for vertical, horizontal in (([], []), ([(np.pi / 6, 20)], []), ([], [(np.pi / 2, 10)])):
            subject = subject_with_lines()
            subject.lines = {"v": vertical, "h": horizontal, "o": []}
            with self.subTest(vertical=vertical, horizontal=horizontal):
                self.assertEqual(subject.clac_intersections(), [])

    def test_multiple_line_pairs_have_expected_coordinates_and_metadata(self):
        subject = subject_with_lines()
        subject.lines["v"] = [(np.pi / 6, 20), (-np.pi / 6, 30)]
        subject.lines["h"] = [(np.pi / 2, 10), (np.pi / 2, 20)]
        results = subject.clac_intersections()
        expected = []
        for vertical in subject.lines["v"]:
            for horizontal in subject.lines["h"]:
                phi, rho = vertical
                y = horizontal[1]
                expected.append(((rho - y * np.sin(phi)) / np.cos(phi), y))
        np.testing.assert_allclose([result.intersection for result in results], expected)
        for result, pair in zip(results, [(v, h) for v in subject.lines["v"] for h in subject.lines["h"]]):
            self.assertEqual((result.line_v, result.line_h), pair)
            self.assertEqual(result.x, (0, 60))

    def test_exact_vertical_line_should_produce_finite_intersection(self):
        subject = subject_with_lines()
        subject.lines["v"] = [(0, 20)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = subject.clac_intersections()[0]
        np.testing.assert_allclose(result.intersection, (20, 10))

    def test_outside_image_intersections_are_currently_kept(self):
        subject = subject_with_lines()
        subject.lines["h"] = [(np.pi / 2, 100)]
        result = subject.clac_intersections()[0]
        self.assertAlmostEqual(result.intersection[1], 100)

    def test_requires_lines_before_computing_intersections(self):
        with self.assertRaises(ValueError):
            Scanner(rectangle_image()).clac_intersections()


class ConnectivityDiagnostics(unittest.TestCase):
    def test_empty_intersections_return_same_empty_list(self):
        subject = Scanner(rectangle_image())
        subject.intersections = []
        self.assertIs(subject.calc_connectivity(), subject.intersections)

    def test_attaches_dilated_image_and_calls_each_intersection(self):
        subject = subject_with_lines()
        subject.intersections = [Mock(), Mock()]
        results = subject.calc_connectivity()
        self.assertIs(results, subject.intersections)
        for result in results:
            self.assertIs(result.image, subject.edges_img_dilated)
            result.connectivity.assert_called_once_with()

    def test_real_intersection_flow_needs_a_sampling_length(self):
        subject = subject_with_lines()
        subject.clac_intersections()
        subject.calc_connectivity()
        for result in subject.intersections:
            np.testing.assert_allclose(result.connectivity(), np.ones(4))

    def test_manually_configured_sampling_length_allows_calculation(self):
        subject = subject_with_lines()
        subject.clac_intersections()
        for result in subject.intersections:
            result.along_length = 4
        results = subject.calc_connectivity()
        np.testing.assert_allclose(results[0].connectivity(), np.ones(4))

    def test_new_dilated_image_should_recompute_existing_connectivity(self):
        subject = subject_with_lines()
        subject.clac_intersections()
        subject.intersections[0].along_length = 4
        result = subject.calc_connectivity()[0]
        np.testing.assert_allclose(result.connectivity(), np.ones(4))
        subject.edges_img_dilated = np.zeros((40, 60), dtype=np.uint8)
        result = subject.calc_connectivity()[0]
        self.assertIs(result.image, subject.edges_img_dilated)
        np.testing.assert_allclose(result.connectivity(), np.zeros(4))

    def test_requires_intersections_before_calculating_connectivity(self):
        with self.assertRaises(ValueError):
            Scanner(rectangle_image()).calc_connectivity()


if __name__ == "__main__":
    unittest.main()
