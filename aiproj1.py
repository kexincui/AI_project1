import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import numpy
import pandas as pd
from pandas import DataFrame
import math

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

#finding Xmin . where X,Y,Z are the points and probability respectively
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


X = np.array(X)
Y = np.array(Y)
xycoordinates = np.column_stack((X, Y))
xy_p = np.column_stack((xycoordinates, Z))
nonzero_xycoordinates = np.column_stack((nonzeroCoor_X, nonzeroCoor_Y))

cord_prob = pd.DataFrame(
    {'Xcoordinate': X,
     'Ycoordinate': Y,
     'Probability': Z
    })

##################Search Algorithm and CDP #####################

final_prob = 0

final_prob_arr = [0]
time_arr =[0]

time = ((Xmin**2+Ymin**2)**0.5)
print("time is :",time)

search_path_X = [0]
search_path_Y = [0]


if ((Xmax-Xmin)%2==0):
	Yend = Ymax 
else: 
	Yend = Ymin
up = True

# plt.ion()
def search_path(startx,starty,up,cord_prob,time,final_prob,final_prob_arr,time_arr):
	if (up is True):# drone going up
		while(starty <= Ymax):
			search_path_X.append(startx)
			search_path_Y.append(starty)
			#To get probability from XY coordinate in dataframe (from pandas)
			time_arr.append(time)
			probability_xy = cord_prob[((cord_prob['Xcoordinate'] == startx) & (cord_prob['Ycoordinate'] == starty))].iat[0,0]
			final_prob += probability_xy
			final_prob_arr.append(final_prob)
			starty += 1
			time += 1	
		starty -= 1
		startx += 1
		up = False
	else:
		while(starty >= Ymin):
			search_path_X.append(startx)
			search_path_Y.append(starty)
			time_arr.append(time)
			probability_xy = cord_prob[((cord_prob['Xcoordinate'] == startx) & (cord_prob['Ycoordinate'] == starty))].iat[0,0]
			final_prob += probability_xy
			final_prob_arr.append(final_prob)
			starty -= 1
			time += 1
		starty += 1
		startx += 1
		up = True
	if (startx > Xmax):
		return
	search_path(startx,starty,up,cord_prob,time,final_prob,final_prob_arr,time_arr)


search_path(Xmin,Ymin,up,cord_prob,time,final_prob,final_prob_arr,time_arr)

final_search_path = np.column_stack((search_path_X, search_path_Y))

search_path_XY = np.column_stack((search_path_X, search_path_Y))

# np.savetxt("heatmap1_path.txt",search_path_XY,fmt='%.1f')
np.savetxt("heatmap2_path.txt",search_path_XY,fmt='%.1f')

fig1 = plt.figure()
fig2 = plt.figure()

ax1 = fig1.add_subplot(111)
ax2 = fig2.add_subplot(111)

ax1.contourf(xi, yi, zi, 15, cmap=plt.cm.rainbow)
# fig1.colorbar()
ax2.plot(time_arr,final_prob_arr)
ax2.grid()
plt.xlabel('Flight time')
plt.ylabel('CDP')
plt.suptitle('CDP v.s. Time')

plt.show()


np.savetxt("finalProb.txt",final_prob_arr,fmt='%.6f')
#################################################

print(final_prob_arr[100])
##################Efficiency EB #####################

# 1. Nr is probability at 100 200 300 600 900
# 2. Sort initial heat map in descending order of probability 
# 3. d is the distance from 0,0 to the closest point of non zero probability

# CDP at T=100 :
# initial_time = int((Xmin**2+Ymin**2)**0.5)
T_arr=[100 , 200, 300, 600, 900]
# T_arr[:] = [x - initial_time for x in T_arr]

Numerator_arr = [] 

for i in range(len(T_arr)):
	Numerator_arr.append(final_prob_arr[T_arr[i]])
print("Numerator is: ",Numerator_arr)
print(final_prob_arr[100])
# finding d
euc_dist_arr =[]

for i in range(len(nonzero_xycoordinates)):
	euc_dist_arr.append(((nonzero_xycoordinates[i][0]**2 + nonzero_xycoordinates[i][1]**2))**0.5)

d = int(min(euc_dist_arr))


# finding T+1-d (upper_limit)
upper_limit_arr =[]

for i in range(len(T_arr)):
	upper_limit_arr.append(T_arr[i]+1-d)

# Sorting the probabilities in descending order(z in descending)
Z_rev=np.sort(Z)[::-1]

denominator = 0
denominator_arr=[]

for j in range(len(upper_limit_arr)):
	for i in range(1,upper_limit_arr[j]+1):
		denominator += Z_rev[i]
	denominator_arr.append(denominator)
	denominator = 0

# for i in range (1,upper_limit_arr[1]+1):
# 	denominator += Z_rev[i]
# denominator_arr.append(denominator)

print("Denominator is: ",denominator_arr)

Efficiency_arr = []

for i in range(len(Numerator_arr)):
	Efficiency_arr.append(Numerator_arr[i]/denominator_arr[i])

print("Efficiency is: ",Efficiency_arr)
cdp_time = np.column_stack((time_arr, final_prob_arr))
np.savetxt("cdp_time.txt",cdp_time,fmt='%.6f')

# print(Efficiency_arr)