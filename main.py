from image_utils import load_image, edge_detection
from PIL import Image
import numpy as np

def main():
    # Load original image
    image = load_image("sonic.png")

    # Apply edge detection
    edges = edge_detection(image)

    # Save result
    edges_uint8 = (edges * 255).astype(np.uint8)
    Image.fromarray(edges_uint8).save("edges.png")

if __name__ == "__main__":
    main()

