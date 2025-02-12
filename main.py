import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import convolve
from image_utils import load_image, edge_detection  # Import functions from image_utils.py
from skimage.filters import median
from skimage.morphology import ball  # Used for noise suppression with ball structuring element



def noise_suppression (image_array):
    clean_image = median(edge_image, ball(5))
    return clean_image  

def binary_image(clean_image, threshold): 
    edge_binary = (clean_image > threshold).astype(np.uint8) 
    return edge_binary 

def save_image (edge_binary):
    edge_image = Image.fromarray((edge_binary * 255).astype(np.uint8)) 
    edge_image.save('my_edges.png') 

image_array = load_image(CHILDWSIDUR.jpg) #creates an image array

clean_image = noise_suppression(image_array) #cleans the array

edge_image = edge_detection(clean_image) #detects edges on the cleaned image

edge_binary = binary_image(edge_image, 'my_edges.png') #

save_image(edge_binary, 'my_edges.png')

plt.imshow(edge_binary, cmap='gray')  
plt.axis('off')  
plt.show()
