from PIL import Image
import numpy as np
from scipy.signal import convolve2d

def load_image(path):
    """
    Load an image and return a 2D grayscale NumPy array
    """
    img = Image.open(path).convert("L")
    image = np.array(img, dtype=np.float32)

    # Ensure image is 2D
    if image.ndim == 3:
        image = image[:, :, 0]

    return image

def edge_detection(image):
    """
    Perform Sobel edge detection and return a normalized 2D array
    """
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

    # Normalize (required by tests)
    if edges.max() != 0:
        edges = edges / edges.max()

    return edges
