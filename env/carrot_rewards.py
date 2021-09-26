import numpy as np
from PIL import Image
import cv2

import torch


def image_transform(image_cv2, size=(32, 32)):
    """
    Convert input image to downsampled binary version.
    input: cv2 image (np.array), shape (500, 500, 3)
    output: cv2 image (np.array), shape (32, 32)
    """
    # 1. Convert image to grayscale.
    image_cv2_grayscale = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
    # 2. Resize. Uses bilinear interpolation by default.
    image_cv2_resized = cv2.resize(image_cv2_grayscale, size, cv2.INTER_AREA)
    # 3. Threshold image to 0-1.
    #_, image_cv2_threshold = cv2.threshold(image_cv2_resized, 0, 255, cv2.THRESH_BINARY)
    # 4. Normalize image between 0~1.
    image_normalized = image_cv2_resized / 255.
    return image_normalized

def lyapunov_measure(tensor):
    """
    Return lyapunov measure by creating a weighted matrix.
    """
    # pixel_radius = 7
    # measure = np.zeros((32, 32))
    # for i in range(32):
    #     for j in range(32):
    #         radius = np.linalg.norm(np.array([i - 15.5, j - 15.5]), ord=2)
    #         measure[i,j] = np.maximum(radius - pixel_radius, 0)
    measure = np.load("env/H3.npy")/100

    if tensor == True:
        measure = torch.FloatTensor(measure).cuda()

    return measure

def lyapunov(image_normalized, tensor=False):
    """
    Apply the lyapunov measure to the image.
    input: cv2 image (np.array), shape (32, 32)
    output: (np.float), shape ()
    """
    V_measure = lyapunov_measure(tensor)
    # element-wise multiplication.

    if len(image_normalized.shape) == 2:
        # image_normalized: 32 x 32
        if tensor == True:
            V = torch.sum(image_normalized * V_measure)
            image_sum = torch.sum(image_normalized)
        else:
            V = np.sum(np.multiply(image_normalized, V_measure))
            image_sum = np.sum(image_normalized)
        r = V / (image_sum +1e-7)

    elif len(image_normalized.shape) == 3:
        # image_normalized: B x 32 x 32
        if tensor == True:
            V = torch.sum(image_normalized * V_measure[None, :, :], dim=(1, 2))
            image_sum = torch.sum(image_normalized, dim=(1, 2))

        else:
            V = np.sum(np.multiply(image_normalized, V_measure[None, :, :]), axis=(1, 2))
            image_sum = np.sum(image_normalized, axis=(1, 2))
        r = V / (image_sum + 1e-7)

    return r
