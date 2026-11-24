"""Frame diagnostics with controlled corners and real scanner integration.

The complete entities module is loaded by the scanner diagnostic fixture.
Only existing import aliases are accommodated; Frame is never replaced or
excluded. Controlled corners isolate Frame from previously reviewed defects
in Intersection, while integration tests use real Intersection objects.
"""

import contextlib
import io
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import warnings

import numpy as np

from test.test_scanner_diagnostics import Scanner, entities


Frame = entities.Frame
RECTANGLE = ((1, 1), (11, 1), (11, 6), (1, 6))


def make_frame(points=RECTANGLE, image_shape=(100, 100), orientations=None,
               line_h=np.pi / 2, line_v=0):
    if orientations is None:
        orientations = [(1, 1, 1, 1)] * 4
    corners = []
    for point, orientation in zip(points, orientations):
        corners.append(SimpleNamespace(
            intersection=point,
            line_h=(line_h, 0),
            line_v=(line_v, 0),
            orientation=Mock(return_value=orientation),
            summary=Mock(return_value=f"Cross at: {point}"),
        ))
    return Frame(*corners, image_shape=image_shape)


def frame_from_scanner():
    subject = Scanner(np.zeros((100, 100), dtype=np.uint8))
    subject.edges_img = np.ones((100, 100), dtype=np.uint8)
    subject.edges_img_dilated = np.ones((100, 100), dtype=np.uint8)
    angle = 0.2
    subject.lines = {
        "v": [(angle, 40), (angle, 60)],
        "h": [(angle + np.pi / 2, 40), (angle + np.pi / 2, 60)],
        "o": [],
    }
    intersections = subject.clac_intersections()
    # Product order is v0h0, v0h1, v1h0, v1h1. Reorder clockwise.
    frame = Frame(intersections[0], intersections[2], intersections[3],
                  intersections[1], image_shape=(100, 100))
    return subject, frame


class FrameGeometryTests(unittest.TestCase):
    def test_constructor_preserves_corners_and_shape(self):
        frame = make_frame()
        self.assertEqual(frame.image_shape, (100, 100))
        self.assertEqual(len(frame.corners), 4)
        self.assertEqual(frame.corners[0].intersection, RECTANGLE[0])

    def test_coordinates_preserve_clockwise_order(self):
        self.assertEqual(make_frame().coordinates(), RECTANGLE)

    def test_axis_aligned_rectangle_area(self):
        self.assertAlmostEqual(make_frame().area(), 50)

    def test_rotated_square_area_is_rotation_invariant(self):
        frame = make_frame(
            points=((10, 0), (20, 10), (10, 20), (0, 10)),
            line_h=3 * np.pi / 4, line_v=np.pi / 4,
        )
        # Adjacent sides have length sqrt(200), and are perpendicular.
        self.assertAlmostEqual(frame.area(), 200)

    def test_unsigned_coordinates_do_not_overflow_differences(self):
        # Use an area above uint8's maximum: a small rectangle can hide the
        # overflow because repeated unsigned arithmetic happens modulo 256.
        points = tuple(tuple(np.uint8(value) for value in point) for point in
                       ((1, 1), (21, 1), (21, 21), (1, 21)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.assertAlmostEqual(make_frame(points=points).area(), 400)

    def test_repr(self):
        self.assertEqual(repr(make_frame()), "rectangle")


class FrameRelativeAreaTests(unittest.TestCase):
    def test_uses_shape_from_constructor(self):
        self.assertAlmostEqual(make_frame().relative_area(), 0.005)

    def test_explicit_current_shape_is_valid(self):
        self.assertAlmostEqual(make_frame().relative_area((100, 100)), 0.005)

    def test_explicit_new_shape_can_be_reused(self):
        frame = make_frame()
        self.assertAlmostEqual(frame.relative_area((200, 100)), 0.0025)
        self.assertEqual(frame.image_shape, (200, 100))
        self.assertAlmostEqual(frame.relative_area((200, 100)), 0.0025)

    def test_missing_shape_needs_explicit_validation(self):
        with self.assertRaises(ValueError):
            make_frame(image_shape=None).relative_area()

    def test_zero_image_area_needs_explicit_validation(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with self.assertRaises(ValueError):
                make_frame(image_shape=(0, 100)).relative_area()

    def test_negative_dimensions_need_explicit_validation(self):
        with self.assertRaises(ValueError):
            make_frame(image_shape=(-100, 100)).relative_area()


class FrameScoreTests(unittest.TestCase):
    def test_score_with_stored_shape(self):
        self.assertAlmostEqual(make_frame().score(), 0.02)

    def test_selects_matching_orientation_index_for_each_corner(self):
        frame = make_frame(orientations=[
            (1, 0, 0, 0), (0, 2, 0, 0), (0, 0, 3, 0), (0, 0, 0, 4),
        ])
        self.assertAlmostEqual(frame.score(), 0.05)

    def test_first_score_with_current_explicit_shape_returns_number(self):
        self.assertEqual(make_frame().score((100, 100)), 0.02)

    def test_score_with_new_shape_updates_and_completes(self):
        frame = make_frame()
        self.assertAlmostEqual(frame.score((200, 100)), 0.01)

    def test_explicit_shape_after_previous_score_remains_valid(self):
        frame = make_frame()
        self.assertAlmostEqual(frame.score(), 0.02)
        self.assertAlmostEqual(frame.score((100, 100)), 0.02)

    def test_unchanged_inputs_reuse_score_cache(self):
        frame = make_frame()
        first = frame.score()
        self.assertEqual(frame.score(), first)
        for corner in frame.corners:
            corner.orientation.assert_called_once_with()


class FrameComparisonTests(unittest.TestCase):
    def test_total_ordering_sorts_different_scores(self):
        low = make_frame(orientations=[(0.5,) * 4] * 4)
        high = make_frame(orientations=[(1,) * 4] * 4)
        self.assertLess(low, high)
        self.assertLessEqual(low, high)
        self.assertGreater(high, low)
        self.assertGreaterEqual(high, low)
        self.assertNotEqual(low, high)
        self.assertEqual(sorted([high, low]), [low, high])

    def test_equal_frames_have_equal_hashes(self):
        first = make_frame()
        translated = make_frame(points=tuple((x + 20, y + 20) for x, y in RECTANGLE))
        self.assertEqual(first, translated)
        self.assertEqual(hash(first), hash(translated))

    def test_set_deduplicates_frames_that_compare_equal(self):
        first = make_frame()
        translated = make_frame(points=tuple((x + 20, y + 20) for x, y in RECTANGLE))
        self.assertEqual(first, translated)
        self.assertEqual(len({first, translated}), 1)

    def test_equality_with_unrelated_type_is_false(self):
        self.assertFalse(make_frame() == object())

    def test_ordering_with_unrelated_type_returns_not_implemented(self):
        self.assertIs(make_frame().__lt__(object()), NotImplemented)


class FrameSummaryTests(unittest.TestCase):
    def test_prints_each_corner_and_area(self):
        frame = make_frame()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertIsNone(frame.summary())
        text = output.getvalue()
        for index, label in enumerate(frame.ORIENTATION_ORDER):
            self.assertIn(f"{label}: Cross at: {RECTANGLE[index]}", text)
            frame.corners[index].summary.assert_called_once_with(one_line=True, println=False)
        self.assertIn("Area: 50", text)


class FrameScannerIntegrationTests(unittest.TestCase):
    def test_whole_entities_module_contains_completed_frame(self):
        self.assertTrue(hasattr(entities, "Frame"))
        self.assertTrue(hasattr(Frame, "__ge__"))
        self.assertTrue(hasattr(Frame, "__le__"))

    def test_frame_from_real_scanner_intersections_has_correct_area(self):
        _, frame = frame_from_scanner()
        # Orthogonal line families, each separated by 20: true area 20*20.
        self.assertAlmostEqual(frame.area(), 400)

    def test_real_connectivity_then_frame_score_is_finite_when_configured(self):
        subject, frame = frame_from_scanner()
        for intersection in subject.intersections:
            intersection.along_length = 3
        subject.calc_connectivity()
        self.assertTrue(np.isfinite(frame.score()))


if __name__ == "__main__":
    unittest.main()
