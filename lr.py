import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#variables to store mean and standard deviation for each feature
mu = []
std = []

def load_data(filename):
	df = pd.read_csv(filename, sep=",", index_col=False)
	df.columns = ["housesize", "rooms", "price"]
	data = np.array(df, dtype=float)
	plot_data(data[:,:2], data[:, -1])
	normalize(data)
	return data[:,:2], data[:, -1]

def plot_data(x, y):
	plt.xlabel('house size')
	plt.ylabel('price')
	plt.plot(x[:,0], y, 'bo')
	plt.show()

def normalize(data):
	for i in range(0,data.shape[1]-1):
		data[:,i] = ((data[:,i] - np.mean(data[:,i]))/np.std(data[:, i]))
		mu.append(np.mean(data[:,i]))
		std.append(np.std(data[:, i]))

def clear_fn(x,theta):
	if x == 1:
		return -1
	if x == 2:
		return -1
	return 0

def h(x,theta):
	return np.matmul(x, theta)


def cost_function(x, y, theta):
	return ((h(x, theta)-y).T@(h(x, theta)-y))/(2*y.shape[0])

def gradient_descent(x, y, theta, learning_rate=0.1, num_epochs=10):
	m = x.shape[0]
	J_all = []
	
	for _ in range(num_epochs):
		h_x = h(x, theta)
		cost_ = (1/m)*(x.T@(h_x - y))
		theta = theta - (learning_rate)*cost_
		J_all.append(cost_function(x, y, theta))
	return theta, J_all 

def plot_cost(J_all, num_epochs):
	plt.xlabel('Epochs')
	plt.ylabel('Cost')
	plt.plot(num_epochs, J_all, 'm', linewidth = "5")
	plt.show()

def test(theta, x):
	x[0] = (x[0] - mu[0])/std[0]
	x[1] = (x[1] - mu[1])/std[1]
	y = theta[0] + theta[1]*x[0] + theta[2]*x[1]
	print("Price of house: ", y)

x,y = load_data("house_price_data.txt")
y = np.reshape(y, (46,1))
x = np.hstack((np.ones((x.shape[0],1)), x))
theta = np.zeros((x.shape[1], 1))
learning_rate = 0.1
num_epochs = 50
theta, J_all = gradient_descent(x, y, theta, learning_rate, num_epochs)
J = cost_function(x, y, theta)
print("Cost: ", J)
print("Parameters: ", theta)

#for testing and plotting cost 
n_epochs = []
jplot = []
count = 0
for i in J_all:
	jplot.append(i[0][0])
	n_epochs.append(count)
	count += 1
jplot = np.array(jplot)
n_epochs = np.array(n_epochs)
plot_cost(jplot, n_epochs)

test(theta, [600, 3])


'''

1. y is present as a list of all the prices, we r converting it into a numpy array
2. x.shape[0] is 46 as x.shape = (46,2) as x has the house prize and no of rooms.
3. create a numpy array of 46 ones. its a 1D array.
4. hstack is used to append matrixes
5. now x would be a numpy array of dimension (46,3) with first column as all ones then the second column as house size and third as no of rooms
6. now x.shape = (46,3) so x.shape[1] = 3
7. so next create a numpy array of 0's with dimension (3,1)
8. our output will be a (3,1) matrix so in order to get that we initilize theta with the same dimension
9. since matrix's should have same no of rows and columns to be multiplied with, we make the x array into a (46,3) dimension instead of (46,2)
10. the hypothesis function does matrix multiplication of x(46,3) and theta(3,1) and the resultant dimension would be (46,1)
11. x would be the house size, no of rooms and all ones matrix, we r doing matrix multiplication of that with an numpy array containing 3 0's 
12. cost function = 1/2m(x*theta-y)^2

'''