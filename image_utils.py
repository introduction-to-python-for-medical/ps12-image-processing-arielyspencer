from PIL import Image
import numpy as np
from scipy.signal import convolve2d

def load_image(image_path):
    try:
        img = Image.open(image_path)
        img_array = np.array(img)
        return img_array
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def edge_detection(image_array):
    grayscale_image = np.mean(image_array, axis=2).astype(np.uint8)

    kernelX = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
    kernelY = np.array([[1, 2, 1],
                       [0, 0, 0],
                       [-1, -2, -1]])

    edgeX = convolve2d(grayscale_image, kernelX, mode='same', boundary='fill', fillvalue=0)
    edgeY = convolve2d(grayscale_image, kernelY, mode='same', boundary='fill', fillvalue=0)

    edgeMAG = np.sqrt(np.square(edgeX) + np.square(edgeY))

    return edgeMAG
