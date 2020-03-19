import numpy as np
from PIL import Image
import os
from scipy.ndimage.filters import convolve
import copy
import sys
import pickle

# Helper functions
def normalize_image(data):
    return data/(np.max(data) + 1e-6)

def threshold_image(data, hold):
    data[(data < hold)] = 0
    return data

def reshape_data(data, x, y):
    return np.reshape(data, (x, y))

# Initialize the Sobel-Feldman filter
X_filter = np.array([[3, 0, -3], [10, 0, -10], [3, 0, -3]])
Y_filter = X_filter.T

# Load Phone Filter
#os.chdir("C:/Users/prudh/Desktop/test/find_phone_task/find_phone")

current_image = 'Patch/44.jpg'
img_file = Image.open(current_image).convert('L')
img_data = np.array(img_file.getdata())
img_data = reshape_data(img_data, img_file.size[1], img_file.size[0])

x_direction = convolve(img_data, X_filter, mode= 'constant')
y_direction = convolve(img_data, Y_filter, mode= 'constant')

# Get gradient
gradient = np.sqrt(np.square(x_direction) + np.square(y_direction))

# Normalize
normalized_gradient = np.uint8(normalize_image(gradient) * 255)

#Calculate threshold
threshold_value = 2 * np.mean(normalized_gradient) + 1.5 * np.std(normalized_gradient)
thresholded = threshold_image(normalized_gradient, threshold_value)

patch = Image.fromarray(thresholded)
patch.save("patch_extractor.png")
