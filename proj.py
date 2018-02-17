import os
import numpy as np
import matplotlib.pyplot as plt
# arr = []
# with open("heatmap_1.txt","r") as f:
# 	lines = f.readlines()
# # print(lines[10]) gives 0 10 0
# type(lines)

with open("heatmap_1.txt","r") as f:
    arr = []
    for line in f:
        line = line.split() # to deal with blank 
        if line:            # lines (ie skip them)
            line = [float(i) for i in line]
            arr.append(line)
# print(arr[3][1]) gives 3.0
# print(len(arr))
arr2=[]
for i in range(len(arr)):
	arr2.append(arr[i][2])
	arr[i]=arr[i][:-1]
# print(arr2)
# print(arr)

plt.xlabel('some others')
plt.ylabel('some numbers')
for i in range(len(arr2)):
	if (arr2[i] > 0):
		plt.plot(arr[i])
plt.show()
