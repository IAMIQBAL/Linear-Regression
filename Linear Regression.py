import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

# Predicting Final Scores based on Number of Hours Study

def loadData(file_name = ''):
	data = pd.read_csv(file_name)
	return data

def fitData(data):
	x = data['Hour']
	y = data['Score']
	x_b = np.c_[np.ones((100, 1)), x]
	return x_b, y.T

def NormalEquation(x, y):

	# theta = (x^T.x).x^T.y
	theta_best = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
	return theta_best

def predict(m, x, b):
	# y = m*x + b
	# x = Number of Hours Study
	y = m*x + b
	return y

def run():
	if __name__ == '__main__':
		data = loadData('data.csv')
		x, y = fitData(data)
		m, b = NormalEquation(x, y)
		hours_of_study = input('Enter the Numbers of Hours Study: ')
		y = predict(b, int(hours_of_study), m)
		print('Predicted Test Scores: ', y)

run()