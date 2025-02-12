from PIL import Image
from skimage.filters import median
from skimage.morphology import disk  # Use disk for 2D images
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
import numpy as np

def load_image(path):
    image = Image.open(path).convert('RGB')  # Ensure image is RGB
    image_array = np.array(image)
    return image_array

def edge_detection(image_array):
    child_mean = np.mean(image_array, axis=2)  # Convert to grayscale
    
    kernelY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])  
    kernelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  
    
    edgeY = convolve(child_mean, kernelY)  
    edgeX = convolve(child_mean, kernelX)  
    
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)  # Corrected power calculation
    
    return edgeMAG

def save_image(edge_image, output_path):
    # Normalize to 0–255 for proper image saving
    edge_image = (edge_image - edge_image.min()) / (edge_image.max() - edge_image.min()) * 255
    edge_image = edge_image.astype(np.uint8)  

    # Convert to image and save
    img = Image.fromarray(edge_image)
    img.save(output_path, "JPEG")  
    print(f"Image saved as {output_path}")

# Load and process image
path = "/content/CHILDWSIDUR.jpg"
output_path = "/content/edge_detected.jpg"  # Change to desired save path

image_array = load_image(path)
edge_image = edge_detection(image_array)

# Save the edge-detected image as a JPG
save_image(edge_image, output_path)

# Display the image
plt.imshow(edge_image, cmap='gray')
plt.axis('off')
plt.show(
