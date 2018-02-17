import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Load data from CSV
dat = np.genfromtxt('heatmap_1.txt', delimiter=' ',skip_header=0)
X_dat = dat[:,0]
Y_dat = dat[:,1]
Z_dat = dat[:,2]

# Convert from pandas dataframes to numpy arrays
X, Y, Z, = np.array([]), np.array([]), np.array([])
for i in range(len(X_dat)):
        X = np.append(X,X_dat[i])
        Y = np.append(Y,Y_dat[i])
        Z = np.append(Z,Z_dat[i])



##################HEAT MAP#####################
# create x-y points to be used in heatmap
xi = np.linspace(X.min(),X.max(),1000)
yi = np.linspace(Y.min(),Y.max(),1000)

# Z is a matrix of x-y values
zi = griddata((X, Y), Z, (xi[None,:], yi[:,None]), method='nearest')

# Create the contour plot
plt.contourf(xi, yi, zi, 15, cmap=plt.cm.rainbow)
plt.colorbar()  
plt.show()
#################################################

