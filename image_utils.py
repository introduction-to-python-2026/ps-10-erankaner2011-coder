from PIL import Image
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

def load_image(path="sonic.png"):
    # Load sonic.png, convert to grayscale, return NumPy array
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.float32)

def edge_detection(image):
    kernel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])

    kernel_y = np.array([[-1, -2, -1],
                         [ 0,  0,  0],
                         [ 1,  2,  1]])

    gx = convolve2d(image, kernel_x, mode="same", boundary="symm")
    gy = convolve2d(image, kernel_y, mode="same", boundary="symm")

    edges = np.sqrt(gx**2 + gy**2)
    return edges

# ---- Run ----
image = load_image()          # uses sonic.png
edges = edge_detection(image)

# Display result
plt.imshow(edges, cmap="gray")
plt.axis("off")
plt.show()
