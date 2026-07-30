import itertools
import cv2
import numpy as np
from skimage.transform import hough_line, hough_line_peaks

class scanner:
    def __init__(self, image: np.ndarray) -> None:
        if len(image.shape) != 2:
            raise ValueError("Image should be a single channel 2D array")
        self.image = image


    def scan(self):
        self.preprocess()


    def preprocess(self, kernel_size = 15, intensity_lower = 0, intensity_upper = 255,
        canny_lower = 10, canny_upper = 70, erode_ks = 3, dilate_ks = 15):

        self.blurred = cv2.medianBlur(self.image, 25)
        self.hist_equalized = cv2.equalizeHist(self.blurred)


        kernel = np.ones((kernel_size, kernel_size), dtype=np.int8)
        self.hist_equalized = cv2.morphologyEx(self.hist_equalized, cv2.MORPH_OPEN, kernel)
        self.hist_equalized = cv2.morphologyEx(self.hist_equalized, cv2.MORPH_CLOSE, kernel)


        mask = cv2.inRange(self.hist_equalized, intensity_lower, intensity_upper)

        
        self.filterred = cv2.bitwise_and(self.hist_equalized, self.hist_equalized, mask=mask)



        edges = cv2.Canny(self.filterred, canny_lower, canny_upper, L2gradient=True, apertureSize=3)
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.edges_img = cv2.drawContours(np.zeros(self.image.shape[0:2]), contours, -1, [128, 255, 0], 3)


        kernel = np.ones((dilate_ks, dilate_ks), dtype=np.int8)
        self.edges_img_dilated = cv2.morphologyEx(self.edges_img, cv2.MORPH_DILATE, kernel)

        kernel = np.ones((erode_ks, erode_ks), dtype=np.int8)
        self.edges_img = cv2.morphologyEx(self.edges_img, cv2.MORPH_ERODE, kernel)



    def hough_transform(self, err=np.pi * 1 / 12, threshold=0.49, ks=3 ):
        theta = np.lispace(-np.pi * 1 / 4,np.pi * 3 /4, 180)
        h, theta, distance = hough_line(self.edges_img, theta)
        hits, phi,  rho = hough_line_peaks(
            h, theta, distance, min_distance=10, min_angle=50, threshold=threshold* h.max(), num_peaks=np.inf)

        lines = list(zip(phi, rho))

        self.lines = dict(v=[], h=[], o=[])
        for ix, (phi, _) in enumerate(lines):
            if abs(phi) < err or abs(phi - np.pi) < err:
                self.lines["v"].append(lines[ix])
            elif abs(phi - np.pi / 2) < err:
                self.lines["h"].append(lines[ix])
            else:
                self.lines["o"].append(lines[ix])

        return self.lines


    def clac_intersections(self):
        combinations = list(itertools.product(self.lines["v"], self.lines["h"]))

        if len(combinations) == 0:
            self.intersections = list()

            return self.intersetions

        pairs = np.array(combinations)
        lines_h = pairs[:, 0, :]
        lines_v = pairs[:, 1, :]

        x = (0, self.edges_img.shape[1])

        # points_h = find_points_on_lines(lines_h, x)
        # points_v = find_points_on_lines(lines_v, x)
        # en desarrollo






