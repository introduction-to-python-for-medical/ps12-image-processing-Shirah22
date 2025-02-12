from PIL import Image
import numpy as np
from scipy.ndimage import convolve

def load_image(path):
    image = Image.open(path)
    image_array = np.array(image)
    return image_array


def edge_detection(image_array):
    child_mean = np.mean(image_array, axis=2)
    
    kernelY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])  
    kernelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  
    
    # Apply the filters to the grayscale image
    edgeY = convolve(child_mean, kernelY)  # Vertical edges
    edgeX = convolve(child_mean, kernelX)  # Horizontal edges
    
    # Calculate edgeMAG
    edgeMAG = np.sqrt(edgeX*2 + edgeY*2)
    
    return edgeMAG
