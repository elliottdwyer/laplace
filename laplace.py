import matlab
import matplotlib
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy import interpolate

width, height = 13, 13 # matrix dimensions
a, b = 6, 6 # center coordinates
r = 6 # radius
    
def function(x, y):
    value = (x-8)**2 - (y+1)**2 # arbitrary function to modify
    return value
    
    
def matrix_initialization():
    
    #initialize the matrices
    force_matrix = np.zeros((width,height))
    
    matrix = np.zeros((width,height))
    ref_matrix = np.zeros((width,height))
    
    for y in range(height):
        for x in range(width):
            # Fill within the boundaries of the circle
            if (x-a)**2 + (y-b)**2 - r**2 < r:
                force_matrix[y,x] = 1
            
            # The following else block maps the points outside the boundary to None.
            # However, this implementation is not supported by the interpolation function interpolate.interp2d(), so we only
            # use it for the plotting onto the coarse-grid unit disk. In that case, those points remain mapped to 0.
            
            #else:
            #    force_matrix[y,x] = None
            
                
            # Map the function on the boundaries
            if abs((x-a)**2 + (y-b)**2 - r**2) < r:

                val = function(x, y)
                force_matrix[y,x] = val
                                
                # Set reference matrix
                ref_matrix[y,x] = 1
            
    
    return ref_matrix, force_matrix, matrix

def pascal(ref_matrix, force_matrix, matrix):
    
    i = 0
    
    while(i < 1000):
        for y in range(height):
            for x in range(width):
                if force_matrix[y,x] > 0:
                    if ref_matrix[y,x] != 1:
                        new_approx = ((force_matrix[y-1,x]) + (force_matrix[y,x-1]) + (force_matrix[y,x+1]) + (force_matrix[y+1,x])) / 4
                        force_matrix[y,x] = new_approx

        i += 1

    # The following lines of code plot the harmonic function on the coarse-grid unit disk

    #plt.imshow(force_matrix, cmap = 'gist_rainbow', interpolation = 'nearest')
    #plt.title("Harmonic Function on Coarse-Grid Unit Disk with Boundary Function f(x,y) = (x-8)^2 - (y+1)^2")
    #plt.colorbar()
    #plt.xlabel("x")
    #plt.ylabel("y")
    #plt.show()
    
    return force_matrix

def interpolation(matrix):
    x = np.linspace(0, width, width)
    y = np.linspace(0, height, height)
    
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
    plt.title("Harmonic Function on Unit Disk with Boundary Function f(x,y) = (x-8)^2 - (y+1)^2")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()
    


def main():
    ref_matrix, force_matrix, matrix = matrix_initialization()
    force_matrix = pascal(ref_matrix, force_matrix, matrix)
    interpolation(force_matrix)
    
    
main()