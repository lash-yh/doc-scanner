import unittest

import numpy as np
import pandas as pd

from src.core.math_utils import (
    _order_points,
    find_points_on_lines,
    find_y_on_lines,
    four_point_transform,
    interpolate_pixels_along_line,
    intersection_cartesian,
    points_to_lines,
)


class LineGeometryTests(unittest.TestCase):
    def test_points_to_line(self):
        result = points_to_lines((1, 2), (3, 6))
        self.assertEqual(result.columns.tolist(), ["A", "B", "C"])
        np.testing.assert_array_equal(result.to_numpy(), [[-4, 2, 0]])

    def test_batched_points_satisfy_their_lines(self):
        p1 = np.array([[2, 0], [0, 3], [1, 1]])
        p2 = np.array([[2, 8], [8, 3], [4, 7]])
        lines = points_to_lines(p1, p2).to_numpy()
        for points in (p1, p2):
            np.testing.assert_allclose(
                lines[:, 0] * points[:, 0]
                + lines[:, 1] * points[:, 1]
                + lines[:, 2],
                0,
            )

    def test_invalid_point_dimensions(self):
        for p1, p2 in [([1], [2]), ([[1, 2, 3]], [[4, 5, 6]]), (1, 2)]:
            with self.subTest(p1=p1, p2=p2), self.assertRaises(ValueError):
                points_to_lines(p1, p2)

    def test_mixed_point_dimensions_are_rejected(self):
        for p2 in ([[3, 4], [5, 6]], [[3], [4]]):
            with self.subTest(p2=p2), self.assertRaises(ValueError):
                points_to_lines([1, 2], p2)

    def test_cartesian_intersections(self):
        # x=2 with y=3; x+y=5 with x-y=1.
        first = pd.DataFrame([[1, 0, -2], [1, 1, -5]], columns=["A", "B", "C"])
        second = pd.DataFrame([[0, 1, -3], [1, -1, -1]], columns=["A", "B", "C"])
        np.testing.assert_allclose(intersection_cartesian(first, second), [(2, 3), (3, 2)])

    def test_intersection_rejects_missing_columns(self):
        valid = pd.DataFrame([[1, 0, -2]], columns=["A", "B", "C"])
        for first, second in [(valid.drop(columns="A"), valid), (valid, valid.drop(columns="C"))]:
            with self.subTest(columns=(list(first), list(second))), self.assertRaises(ValueError):
                intersection_cartesian(first, second)

    def test_sample_horizontal_and_diagonal_lines(self):
        lines = [(np.pi / 2, 3), (np.pi / 4, np.sqrt(2))]
        np.testing.assert_allclose(find_y_on_lines(lines, [0, 1, 2]), [[3, 3, 3], [2, 1, 0]], atol=1e-12)

    def test_sample_single_polar_line(self):
        np.testing.assert_allclose(find_y_on_lines((np.pi / 2, 5), [0, 10]), [[5, 5]])

    def test_empty_polar_lines(self):
        self.assertEqual(find_y_on_lines([], [0, 1]), [])
        self.assertEqual(find_points_on_lines([], [0, 1]), [])

    def test_invalid_polar_dimensions(self):
        for lines in ([1, 2, 3], [[1, 2, 3]], np.zeros((2, 2, 2))):
            with self.subTest(lines=lines), self.assertRaises(ValueError):
                find_y_on_lines(lines, [0, 1])

    def test_sample_points_satisfy_polar_equation(self):
        lines = np.array([[np.pi / 2, 3], [np.pi / 4, 2]])
        samples = find_points_on_lines(lines, [0, 4, 8])
        self.assertEqual(len(samples), 3)
        for x, points in zip([0, 4, 8], samples):
            points = np.array(points)
            np.testing.assert_allclose(points[:, 0], x)
            np.testing.assert_allclose(
                points[:, 0] * np.cos(lines[:, 0]) + points[:, 1] * np.sin(lines[:, 0]),
                lines[:, 1],
            )


class PixelInterpolationTests(unittest.TestCase):
    def test_horizontal_vertical_and_diagonal_segments(self):
        for end in [(5, 0), (0, 5), (5, 5), (5, -5)]:
            with self.subTest(end=end):
                pixels = interpolate_pixels_along_line((0, 0), end)
                self.assertIn((0, 0), pixels)
                self.assertIn(end, pixels)
                self.assertTrue(all(isinstance(value, np.integer) for point in pixels for value in point))

    def test_reversing_endpoints_preserves_pixels(self):
        for start, end in [((1, 2), (8, 5)), ((2, 8), (5, 1))]:
            with self.subTest(start=start, end=end):
                self.assertEqual(
                    interpolate_pixels_along_line(start, end),
                    interpolate_pixels_along_line(end, start),
                )

    def test_numpy_endpoints(self):
        self.assertEqual(
            interpolate_pixels_along_line(np.array([1, 2]), np.array([5, 4])),
            interpolate_pixels_along_line((1, 2), (5, 4)),
        )

    def test_larger_width_includes_narrower_segment(self):
        narrow = set(interpolate_pixels_along_line((0, 0), (5, 0), width=1))
        wide = set(interpolate_pixels_along_line((0, 0), (5, 0), width=3))
        self.assertLess(narrow, wide)

    def test_invalid_endpoint_length(self):
        with self.assertRaises(ValueError):
            interpolate_pixels_along_line((1, 2, 3), (4, 5))

    def test_zero_length_segment_contains_its_pixel(self):
        self.assertIn((2, 3), interpolate_pixels_along_line((2, 3), (2, 3)))


class PerspectiveTests(unittest.TestCase):
    def test_corner_order(self):
        result = _order_points([(8, 6), (1, 6), (8, 2), (1, 2)])
        np.testing.assert_array_equal(result, [(1, 2), (8, 2), (8, 6), (1, 6)])
        self.assertEqual(result.dtype, np.float32)

    def test_transform_maps_all_four_corners(self):
        image = np.arange(100, dtype=np.uint8).reshape(10, 10)
        result = four_point_transform(image, [(8, 7), (2, 2), (2, 7), (8, 2)])
        self.assertEqual(result.shape, (5, 6))
        for target, source in [((0, 0), (2, 2)), ((0, -1), (2, 8)), ((-1, -1), (7, 8)), ((-1, 0), (7, 2))]:
            self.assertEqual(result[target], image[source])

    def test_transform_preserves_color_channels(self):
        image = np.full((10, 10, 3), [20, 80, 160], dtype=np.uint8)
        result = four_point_transform(image, [(1, 1), (8, 1), (8, 6), (1, 6)])
        self.assertEqual(result.shape, (5, 7, 3))
        np.testing.assert_array_equal(result, np.broadcast_to([20, 80, 160], result.shape))
