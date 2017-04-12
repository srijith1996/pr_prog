import numpy as np

filename = "/home/ashwin/m"
no_of_data_points = 8125
i= 0
f = open(filename, 'r')
data = [[]]*no_of_data_points
for line in f:
    listWords = line.split(" ")
    data[i] = np.append(data[i],listWords)
    i = i+1
for row in range(len(data)):
	for column in range(len(data[row])):
		data[row][column] = data[row][column].replace(":1","")
for row in range(len(data)):
	data[row] = data[row][:-1]
	data[row] = map(int, data[row])
data = np.array(data)
mins = np.amin(data, axis=0)
print mins



