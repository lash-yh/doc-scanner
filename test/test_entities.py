"""Behavioral checks and isolated diagnostics for Intersection.

The temporary import alias excludes the already reviewed import problem.
Diagnostic tests bypass constructor/attribute blockers explicitly so that one
early exception does not hide independent numerical and cache defects.
"""

import contextlib
import importlib
import io
import sys
import unittest
import warnings
from unittest.mock import patch

import numpy as np

from src.core import math_utils


with patch.dict(sys.modules, {"math_utils": math_utils}):
    entities = importlib.import_module("src.core.entities")

Intersection = entities.Intersection


def make_intersection(**kwargs):
    # A tilted vertical line avoids the vertical-line singularity. Two sample
    # abscissas avoid the scalar-width blocker in the constructor.
    return Intersection(
        (10, 10),
        (np.pi / 6, 10 * np.cos(np.pi / 6) + 5),
        (np.pi / 2, 10),
        **{"x": (0, 20), **kwargs},
    )


def make_sampling_subject():
    subject = make_intersection(along_length=3, width=1)
    subject.image = np.ones((21, 21), dtype=np.uint8)
    # Bypass the independently tested self.intersections typo, and use exact
    # axis-aligned rays to isolate sampling from corner construction.
    subject.intersections = subject.intersection
    subject.corners = ((4, 10), (16, 10), (10, 4), (10, 16))
    return subject


class ConstructionTests(unittest.TestCase):
    def test_accepts_two_sample_abscissas(self):
        subject = make_intersection()
        self.assertEqual(subject.intersection, (10, 10))
        self.assertEqual(len(subject.corners), 4)
        self.assertTrue(np.isfinite(subject.corners).all())

    def test_corners_lie_on_their_lines(self):
        subject = make_intersection()
        for points, line in ((subject.corners[:2], subject.line_h),
                             (subject.corners[2:], subject.line_v)):
            points = np.asarray(points)
            phi, rho = line
            np.testing.assert_allclose(
                points[:, 0] * np.cos(phi) + points[:, 1] * np.sin(phi), rho,
            )

    def test_corners_follow_declared_order(self):
        subject = make_intersection()
        # ORDER names the public connectivity tuple: top, right, bottom, left.
        left, right, top, bottom = subject.corners
        actual = dict(zip(subject.ORDER, subject.corners))
        expected = {"top": top, "right": right, "bottom": bottom, "left": left}
        self.assertEqual(actual, expected)

    def test_accepts_scalar_width(self):
        subject = make_intersection(x=20)
        self.assertTrue(np.isfinite(subject.corners).all())

    def test_accepts_numpy_image(self):
        image = np.zeros((20, 30), dtype=np.uint8)
        subject = make_intersection(image=image)
        self.assertIs(subject.image, image)

    def test_axis_aligned_vertical_line_has_finite_corners(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            subject = Intersection((10, 10), (0, 10), (np.pi / 2, 10), x=(0, 20))
        self.assertTrue(np.isfinite(subject.corners).all(), subject.corners)

    def test_rejects_invalid_coordinate_and_line_lengths(self):
        for point, vertical, horizontal in (
            ((1,), (0, 2), (1, 3)),
            ((1, 2), (0,), (1, 3)),
            ((1, 2), (0, 2), (1,)),
        ):
            with self.subTest(point=point, vertical=vertical, horizontal=horizontal):
                with self.assertRaises(ValueError):
                    Intersection(point, vertical, horizontal, x=(0, 20))

    def test_requires_image_or_abscissas(self):
        with self.assertRaises(ValueError):
            Intersection((10, 10), (0, 10), (np.pi / 2, 10))

    def test_repr(self):
        self.assertEqual(repr(make_intersection()), "Cross at: (10.00, 10.00)")


class PublicConnectivityTests(unittest.TestCase):
    def test_connectivity_uses_stored_intersection(self):
        subject = make_intersection(along_length=3, width=1)
        # Attaching an image later is also how scanner.calc_connectivity works.
        subject.image = np.ones((21, 21), dtype=np.uint8)
        np.testing.assert_allclose(subject.connectivity(), np.ones(4))


class ConnectivityDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.subject = make_sampling_subject()

    def test_full_image_returns_full_connectivity(self):
        np.testing.assert_allclose(self.subject.connectivity(), np.ones(4))

    def test_blank_image_returns_zero_connectivity(self):
        self.subject.image.fill(0)
        np.testing.assert_allclose(self.subject.connectivity(), np.zeros(4))

    def test_samples_xy_as_image_yx(self):
        self.subject.image.fill(0)
        self.subject.image[2, 7] = 1
        with patch.object(entities, "interpolate_pixels_along_line", return_value=[(7, 2)]):
            np.testing.assert_allclose(self.subject.connectivity(), np.ones(4))

    def test_negative_coordinates_do_not_wrap_around_image(self):
        # NumPy accepts -1, but a ray outside the image must not sample the
        # opposite border. Include one valid dark sample to avoid 0/0 here.
        self.subject.image.fill(0)
        self.subject.image[2, -1] = 1
        self.subject.image[-1, 2] = 1
        with patch.object(entities, "interpolate_pixels_along_line",
                          return_value=[(-1, 2), (2, -1), (3, 3)]):
            np.testing.assert_allclose(self.subject.connectivity(), np.zeros(4))
            self.assertEqual(self.subject.longth, [1] * 4)

    def test_no_valid_samples_produce_finite_zero_connectivity(self):
        with patch.object(entities, "interpolate_pixels_along_line", return_value=[(100, 100)]):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                np.testing.assert_array_equal(self.subject.connectivity(), np.zeros(4))

    def test_missing_image_is_rejected_explicitly(self):
        self.subject.image = None
        with self.assertRaises(ValueError):
            self.subject.connectivity()

    def test_missing_along_length_is_rejected_explicitly(self):
        self.subject.along_length = None
        with self.assertRaises(ValueError):
            self.subject.connectivity()

    def test_rejects_nonpositive_sampling_parameters(self):
        for parameters in ({"along_length": 0}, {"along_length": -2},
                           {"width": 0}, {"width": -1}):
            with self.subTest(parameters=parameters):
                subject = make_sampling_subject()
                with self.assertRaises(ValueError):
                    subject.connectivity(**parameters)

    def test_identical_parameters_reuse_connectivity(self):
        first = self.subject.connectivity()
        with patch.object(entities, "interpolate_pixels_along_line") as interpolate:
            self.assertIs(self.subject.connectivity(along_length=3, width=1), first)
            interpolate.assert_not_called()

    def test_changed_width_recomputes_connectivity_and_orientation(self):
        self.subject.image.fill(0)
        self.subject.image[1, 1] = 1
        with patch.object(entities, "interpolate_pixels_along_line",
                          side_effect=[[(1, 1)]] * 4 + [[(2, 2)]] * 4):
            np.testing.assert_allclose(self.subject.orientation(), [2] * 4)
            np.testing.assert_allclose(self.subject.connectivity(width=2), np.zeros(4))
            np.testing.assert_allclose(self.subject.orientation(), np.zeros(4))


class OrientationAndSummaryTests(unittest.TestCase):
    def test_opposite_rays_do_not_form_a_corner(self):
        # Actual corners are left, right, top, bottom. Either opposite pair
        # must contribute one ray to each adjacent corner, never a score of 2.
        for connectivity in ((1, 1, 0, 0), (0, 0, 1, 1)):
            with self.subTest(connectivity=connectivity):
                subject = make_intersection()
                with patch.object(subject, "connectivity", return_value=connectivity):
                    np.testing.assert_allclose(subject.orientation(), np.ones(4))

    def test_summary_can_return_or_print_statistics(self):
        subject = make_intersection()
        subject.hits = [1, 2, 3, 4]
        subject.longth = [4] * 4
        with patch.object(subject, "connectivity", return_value=(0.25, 0.5, 0.75, 1)), \
             patch.object(subject, "orientation", return_value=[1, 1.25, 1.5, 1.75]):
            multiline = subject.summary(println=False)
            self.assertIn("Cross at: (10.00, 10.00)\n", multiline)
            self.assertIn("0.2500(1/4)", multiline)
            self.assertIn("Orientation scores:1.0000, 1.2500, 1.5000, 1.7500,", multiline)
            one_line = subject.summary(one_line=True, println=False)
            self.assertNotIn("\n", one_line)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertIsNone(subject.summary())
            self.assertEqual(output.getvalue(), multiline + "\n")


if __name__ == "__main__":
    unittest.main()
