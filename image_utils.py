
from PIL import Image
import numpy as np
from scipy.signal import convolve2d

def load_image(path):
    # Open image, convert to grayscale, return as NumPy array
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.float32)

def edge_detection(image):
    # Sobel kernels
    kernel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])

    kernel_y = np.array([[-1, -2, -1],
                         [ 0,  0,  0],
                         [ 1,  2,  1]])

    # Convolve image with kernels
    gx = convolve2d(image, kernel_x, mode="same", boundary="symm")
    gy = convolve2d(image, kernel_y, mode="same", boundary="symm")

    # Gradient magnitude
    edges = np.sqrt(gx**2 + gy**2)

    return edges
