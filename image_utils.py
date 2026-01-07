from PIL import Image
import numpy as np
from scipy.signal import convolve2d

def load_image(path):
    """
    Load image as grayscale but KEEP a channel dimension (H, W, 1)
    Required for median(image, ball(3)) in the tests.
    """
    img = Image.open(path).convert("L")
    image = np.array(img, dtype=np.float32)

    # Add channel dimension → (H, W, 1)
    if image.ndim == 2:
        image = image[:, :, np.newaxis]

    return image

def edge_detection(image):
    """
    Perform Sobel edge detection.
    Accepts (H, W, 1) and returns a 2D binary-compatible edge map.
    """
    # Remove channel dimension for convolution
    if image.ndim == 3:
        image = image[:, :, 0]

    kernel_x = np.array([
        [-1,  0,  1],
        [-2,  0,  2],
        [-1,  0,  1]
    ], dtype=np.float32)

    kernel_y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float32)

    gx = convolve2d(image, kernel_x, mode="same", boundary="symm")
    gy = convolve2d(image, kernel_y, mode="same", boundary="symm")

    edges = np.sqrt(gx ** 2 + gy ** 2)

    return edges
