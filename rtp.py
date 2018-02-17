import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

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
for i in range(len(Z)):
	if Z[i] > 0:
		nonzeroCoor_X.append(X[i])
		nonzeroCoor_Y.append(Y[i])

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

for j in range(int(Xmin),int(Xmax)+1):
	for i in range(int(Ymin),int(Ymax)+1):
		search_path_X.append(j)
		search_path_Y.append(i)


# print(search_path_Y)
# startx=Xmin
# starty=Ymin
# 'Yes' if fruit == 'Apple' else 'No'
if ((Xmax-Xmin)%2==0):
	Yend = Ymax 
else: 
	Yend = Ymin
up = True


def search_path(startx,starty,up):
	if (up==True):# drone going up
		while(starty <= Ymax):
			search_path_X.append(startx)
			search_path_Y.append(starty)
			starty += 1
		starty -= 1
		startx += 1
		up = False
	else:
		while(starty >= Ymin):
			search_path_X.append(startx)
			search_path_Y.append(starty)
			
			starty -= 1
		starty += 1
		startx += 1
		up = True
	if (starty == Yend):
		return
	search_path(startx,starty,up)

# print(search_path_Y)

search_path(Xmin,Ymin,up)

plt.scatter(search_path_X,search_path_Y)
plt.show()