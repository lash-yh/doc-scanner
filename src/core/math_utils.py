from typing import Union

import cv2
import numpy as np
import pandas as pd


def intersection_cartesian(L1: pd.DataFrame, L2: pd.DataFrame):

    if not ["A", "B","C"].issubset(set(L1.columns)) or not ["A","B","C"].issubset(set(L2.columns)):
        raise ValueError("L1 and L2 should both contains columns A, B and C, which depicts lines in general form")

    d = L1["A"] * L2["B"] - L1["B"] * L2["A"]
    dx = L1["B"] * L2["C"] - L1["C"] * L2["B"]
    dy = L1["C"] * L2["A"] - L1["A"] * L2["C"]

    x = dx / d
    y = dy / dx

    return list(zip(x.values.tolist(), y.values.tolist()))



def points_to_lines(p1,p2):
    p1 = np.array(p1)
    p2 = np.array(p2)

    if p1.dtype == np.object_ or p2.dtype == np.object_:
        raise ValueError("p1 and p2 should matrix alike")
    elif len(p1.shape) == 2 and len(p2.shape) == 2:
        if p1.shape[1] != 2 or p2.shape[1] != 2:
            raise ValueError("p1 and p2 should be a matrix with column size of exactly 2")
    elif len(p1.shape) == 1 and len(p1) == 2 and len(p1.shape) == 1 and len(p2) == 2:
        p1 = p1.reshape(-1, 2)
        p2 = p2.reshape(-1, 2)
    else:
        raise ValueError("Invalid p1 and p2")

    a = p1[:, 1] - p2[:, 1]
    b = p2[:, 0] - p1[:, 1]
    c = p1[:, 0] * p2[:, 1] - p2[:,0] * p1[:, 1]

    return pd.DataFrame([a,b,c], index=["A", "B", "C"]).T


def find_y_on_lines(lines: np.array, x: np.array):
    if len(lines) == 0:
        return lines
    lines = np.array(lines)
    if lines.dtype == np.object_:
        raise ValueError("lines should be matriz alike")
    elif len(lines.shape) ==1:
        if len(lines) == 2:
            lines = lines.reshape(-1, 2)
        else:
            raise ValueError("the length of line vector should 2")
    elif len(lines.shape) == 2:
        if lines.shape[1] != 2:
            raise ValueError("lines should be matrix with column size of exactly 2")
    else:
        raise ValueError("Invalid lines")


    x = np.array(x)
    if x.dtype == np.object_:
        raise ValueError("x should be matrix alike")

    rho = lines[:, 1].reshape(-1,1)
    phi = lines[:, 0].reshape(-1,1)
    y = (rho - x * np.cos(phi)) / np.sin(phi)

    return y


def find_points_on_lines(lines: np.array, x: np.array):
    if len(lines) == 0:
        return lines

    lines = np.array(lines)
    if len(lines.shape) == 1:
        if len(lines) == 2:
            lines = lines.reshape(-1, 2)

    x = np.array(x)

    y = find_y_on_lines(lines, x)

    points = list()

    for ix in range(len(x)):
        points_on_a_line = np.zeros((len(lines), 2))
        points_on_a_line[:, 0] = x[ix]
        points_on_a_line[:, 1] = y[:, ix]
        points.append(list(map(lambda x: tuple(x), points_on_a_line.tolist())))

    return points


def interpolate_pixels_along_line():
    pass

