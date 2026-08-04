from functools import total_ordering

import numpy as np

from math_utils import find_points_on_lines, interpolate_pixels_along_line

class Intersection:
    ORDER = ["top", "right", "bottom", "left"]
    ORIENTATION_ORDER = ["top-left", "top-right", "bottom-right", "bottom-left"]

    def __init__(
            self, intersection, line_v, line_h, x=None, image=None, along_length=None, width=3
    ):

        if len(intersection) != 2:
            raise ValueError("intersection must be a tuple like (x, y)")
        elif len(line_v) != 2:
            raise ValueError("line_v must be a tuple like (phi, rho)")
        elif len(line_h) != 2:
            raise ValueError("line_h must be a tuple like (phi, rho)")

        self.intersection = intersection
        self.line_v = line_v
        self.line_h = line_h
        self.__connectivity = None
        self.__orientation = None
        self.image = image
        self.along_length = along_length
        self.width = width