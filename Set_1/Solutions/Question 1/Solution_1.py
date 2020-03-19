import numpy as np

# load data
input_numbers = open("random_numbers.txt").read().splitlines()
input_numbers = np.array(input_numbers, dtype =np.float32)

def without_inbuilt_function(input_numbers):
    tanh_numerator = np.exp(2*input_numbers)-1
    tanh_denominator = np.exp(2*input_numbers)+1
    tanh_output = tanh_numerator/tanh_denominator
    return tanh_output

def with_inbuilt_function(input_numbers):
    return np.tanh(input_numbers)

def sanity_check_output(output_values):
    assert np.max(output_values) <=1 and np.min(output_values) >=-1
 
