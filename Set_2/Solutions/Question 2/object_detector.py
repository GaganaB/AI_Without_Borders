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

current_image = '/Testing_Images/90.jpg'
img_file = Image.open(current_image).convert('L')
img_data = np.array(img_file.getdata())
img_data = reshape_data(img_data, img_file.size[1], img_file.size[0])

filter_file = Image.open('/Patch/f1.jpg').convert('L')
filtered_data = np.array(filter_file.getdata())
fil = reshape_data(fil_data, filter_file.size[1], filter_file.size[0])
   
# Apply Sobel-Feldman filter across both directions
x_direction = convolve(img_data, X_filter, mode= 'constant')
y_direction = convolve(img_data, Y_filter, mode= 'constant')

# Get gradient
gradient = np.sqrt(np.square(x_direction) + np.square(y_direction))

# Normalize
normalized_gradient = np.uint8(normalize_image(gradient) * 255)

#Calculate threshold
threshold_value = 2 * np.mean(normalized_gradient) + 1.5 * np.std(normalized_gradient)
thresholded = threshold_image(normalized_gradient, threshold_value)

# Copy the thresholded image
img = copy.deepcopy(thresholded)

# Convolve phone kernel and the thresholded image    
convolution = convolve(normalize_image(img), normalize_image(fil))

# Normalize
normalized_output = np.uint8(normalize_image(convolution) * 255)

# Calculate threshold
output_threshold_value = np.mean(normalized_output) + np.std(normalized_output)
thresholded_output_image = threshold_image(normalized_output, output_threshold_value)

# Find maximum activation location
max_activation_position = np.argwhere(thresholded_output_image== np.max(thresholded_output_image))
chosen_position = max_activation_position[0]
   
# Convert the retrieved indexes to required scale and round them
normy, normx = chosen_position[0]/float(img_file.size[1]), chosen_position[1]/float(img_file.size[0])
normy, normx = round(normy, 4), round(normx, 4)    

print normx, normy
