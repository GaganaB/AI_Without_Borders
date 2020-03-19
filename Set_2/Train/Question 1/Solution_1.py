import numpy as np

# load data
input_numbers = open("random_numbers.txt").read().splitlines()
input_numbers = np.array(input_numbers, dtype =np.float32)

def without_inbuilt_function(input_numbers):
    sigmoid_output = 1/(1 + np.exp(-x)) 
    return sigmoid_output

def sanity_check_output(output_values):
    assert np.max(output_values) <=1 and np.min(output_values) >=0
