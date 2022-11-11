import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from scipy import interpolate
from scipy.integrate import quad

width, height = 13, 13 # matrix dimensions
a, b = 6, 6 # center coordinates
r = 6 # radius
j = 10 # number of relaxations
    
def function(x, y):
    value = x**2 # arbitrary function to modify
    return value
    
    
def error(matrix):
    exp_value = matrix[6,6]
    exact_value = (1/(2*math.pi)*quad(integrand, 0, 2*math.pi)[0])
    error = abs(exp_value - exact_value)
    mmax = matrix.max()
    mmin = matrix.min()
    error = error / abs(mmax - mmin)
    return error

def integrand(t):
    return function(math.cos(t),math.sin(t))
    
def matrix_initialization():
    
    exact_value = (1/(2*math.pi)*quad(integrand, 0, 2*math.pi)[0])
    
    #initialize the matrices
    force_matrix = np.zeros((width,height))
    
    ref_matrix = np.zeros((width,height))
    
    for y in range(height):
        for x in range(width):
            # Fill within the boundaries of the circle
            if (x-a)**2 + (y-b)**2 - r**2 < r:
                force_matrix[y,x] = exact_value
            else:
                force_matrix[y,x] = None
            
                
            # Map the function on the boundaries
            if abs((x-a)**2 + (y-b)**2 - r**2) < r:

                val = function(x, y)
                force_matrix[y,x] = val
                                
                # Set reference matrix
                ref_matrix[y,x] = 1
            
    
    return ref_matrix, force_matrix

def pascal(ref_matrix, force_matrix, j):
    
    i = 0
    while(i < j):
        for y in range(height):
            for x in range(width):
                if force_matrix[y,x] > 0:
                    if ref_matrix[y,x] != 1:
                        new_approx = ((force_matrix[y-1,x]) + (force_matrix[y,x-1]) + (force_matrix[y,x+1]) + (force_matrix[y+1,x])) / 4
                        force_matrix[y,x] = new_approx

        i += 1
        
    # The following lines of code plot the harmonic function on the coarse-grid unit disk
    
    fig, ax = plt.subplots()
    plt.imshow(force_matrix, cmap = 'gist_rainbow', interpolation = 'nearest')
    plt.suptitle("Harmonic Function on Coarse-Grid Unit Disk with f(x,y) = sin(xy)")
    plt.title("Number of Relaxations: " + str(j), fontsize = 12)
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    
    plt.show()
    
    return force_matrix


def interpolation(matrix, j):
    
    # Replace all "None" cells with 0's to support use of interp2d function
    
    for y in range(height):
        for x in range(width):
            if np.isnan(matrix[y,x]):
                matrix[y,x] = 0
    
    x = np.linspace(0, width, width)
    y = np.linspace(0, height, height)
    
    err = error(matrix)
    
    f = interpolate.interp2d(x, y, matrix, kind = 'linear')
    
    xv = np.linspace(0, 12, 1000)
    yv = np.linspace(0, 12, 1000)
    
    theta = np.linspace(0, 2*np.pi, 400)
    center, radius = [0.5, 0.5], 0.45
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = matplotlib.path.Path(verts * radius + center)
    fig, ax = plt.subplots()
    
    plt.pcolormesh(xv, yv, f(xv, yv), clip_path=(circle, ax.transAxes), cmap = 'gist_rainbow')
    plt.colorbar()
    plt.title("Error at center with respect to range: " + str(round(err,5))+ ";   Number of Relaxations: " + str(j), fontsize = 12)
    plt.suptitle("Harmonic Function on Unit Disk with f(x,y) = x**2\n")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()
    

def main(j):
    ref_matrix, force_matrix = matrix_initialization()
    force_matrix = pascal(ref_matrix, force_matrix, j)
    interpolation(force_matrix, j)
    
main(j)
    