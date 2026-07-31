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
            raise ValueError()



def find_points_on_lines():
    pass

