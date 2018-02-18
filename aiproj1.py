import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import numpy
import pandas as pd
# from pandas import dataframe  
# Load data from CSV
dat = np.genfromtxt('heatmap_2.txt', delimiter=' ',skip_header=0)
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
# plt.contourf(xi, yi, zi, 15, cmap=plt.cm.rainbow)
# plt.colorbar()  
# plt.show()
#################################################


##################Bounding Area#####################

#finding Xmin . where X,Y,Z are the points and probability resp
nonzeroCoor_X = []
nonzeroCoor_Y = []
nonzeroProb_Z = []
index = []
for i in range(len(Z)):
	if Z[i] > 0:
		nonzeroCoor_X.append(X[i])
		index.append(i)
		nonzeroCoor_Y.append(Y[i])
		nonzeroProb_Z.append(Z[i])

# The area is Xmin, Xmax, Ymin, Ymax
bounding_area = []
Xmin=min(nonzeroCoor_X)
Xmax=max(nonzeroCoor_X)
Ymin=min(nonzeroCoor_Y)
Ymax=max(nonzeroCoor_Y)
bounding_area.append(Xmin)
bounding_area.append(Xmax)
bounding_area.append(Ymin)
bounding_area.append(Ymax)

# print(bounding_area)
#################################################





##################Search Algorithm#####################
search_path_X = [0]
search_path_Y = [0]


if ((Xmax-Xmin)%2==0):
	Yend = Ymax 
else: 
	Yend = Ymin
up = True

plt.ion()
def search_path(startx,starty,up,Z):
	if (up is True):# drone going up
		while(starty <= Ymax):
			search_path_X.append(startx)
			search_path_Y.append(starty)
			starty += 1
			# plt.scatter(startx,starty)
			# plt.show()
			# plt.pause(0.0001)
		starty -= 1
		startx += 1
		up = False
	else:
		while(starty >= Ymin):
			search_path_X.append(startx)
			search_path_Y.append(starty)
			# plt.scatter(startx,starty)
			# plt.show()
			# plt.pause(0.0001)
			starty -= 1

		starty += 1
		startx += 1
		up = True
	if (startx==Xmax):
		return
	search_path(startx,starty,up,Z)


search_path(Xmin,Ymin,up,Z)

final_search_path = np.column_stack((search_path_X, search_path_Y))
#################################################


##################CDP#####################


X = np.array(X)
Y = np.array(Y)
xycoordinates = np.column_stack((X, Y))
xy_p = np.column_stack((xycoordinates, Z))


# creating a 2d list with non zero probabilities
nonzero_xy_p = xy_p[xy_p[:, 2] != 0.0]
print (nonzero_xy_p)


final_prob=0
time = 1
final_prob_arr = []
time_arr =[]

def finding_cdp(nonzero_xy_p,final_prob_arr,time_arr,final_prob,time):

	for i in range(len(nonzero_xy_p)):
		final_prob += nonzero_xy_p[i][2]
		time += 1
		final_prob_arr.append(final_prob)
		time_arr.append(time)

finding_cdp(nonzero_xy_p,final_prob_arr,time_arr,final_prob,time)
plt.scatter(time_arr,final_prob_arr)
plt.show(block=True)

	# if (up is True):# drone going up
	# 	while(starty <= Ymax):
	# 		starty += 1
	# 	starty -= 1
	# 	startx += 1
	# 	up = False
	# else:
	# 	while(starty >= Ymin):
	# 		search_path_X.append(startx)
	# 		search_path_Y.append(starty)
	# 		starty -= 1

	# 	starty += 1
	# 	startx += 1
	# 	up = True
	# if (startx==Xmax):
	# 	return
	# finding_cdp(nonzero_xy_p[0][0],starty[0][1],nonzero_xy_p,up,final_prob)

# finding_cdp(mins_x,mins_y,xy_p,up)
##############################################