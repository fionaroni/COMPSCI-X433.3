#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 17:33:40 2018

@author: Fiona Tang
"""

import numpy as np
from numpy import array, matrix, random

# Create a matrix A with shape (3,5) containing random numbers
A = matrix(np.random.random(15))
A = A.reshape(3,5)

# Find size, shape, and length of matrix A
A.size
A.shape
len(A)

# Resize matrix A to (3,4)
A = A.flat[:-3].reshape(3,4)

# Find the transpose of matrix A and assign it to B
B = A.T

# Find the min value in column 1 (not column 0) of matrix B
B[:,1].min()

# Find the min and max values for the entire matrix A
A.min()
A.max()

# Create vector X (an array) with 4 random numbers
X=array(np.random.random(4))

# Create a function and pass vector X and matrix A to it
def multiply(matrix, vector):
    new_vect=vector.reshape(4,1) # changes 1D array to 2D array with shape (4,1)
    return matrix*new_vect

# Alternative function implementation
def multiply2(matrix, vector):
    new_vect=vector[np.newaxis].T # changes 1D array to 2D array with shape (4,1)
    return matrix*new_vect

# Multiply vector X with matrix A and assign the result to D
D = multiply(A,X) # 3x4 matrix multiplied by 4x1 matrix results in 3x1 matrix

# Create a complex number Z with absolute and real parts != 0
Z = 3+4j

# Show its real and imaginary parts as well as it's absolute value
Z.real
Z.imag
abs(Z)

# Multiply result D with the absolute value of Z, and record it to C
C=D*Z

# Convert matrix B from a matrix to a string and overwrite B
B=str(B)

# Display a text on the screen
print("Fiona Tang is done with HW2")





