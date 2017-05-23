
'''
Aidan Holloway-Bidwell and Allie Warren
Data Mining
Final Project
Decision Trees
'''

import numpy as np
import random as r

'''
A Node class for our decision tree. Left child contains subtree with attribute
values less than or equal to the split value, and right child contains subtree
with attribute values greater than split value
'''
class Node:

	def __init__(self, attribute, split_value, left=None, right=None):
		self.attribute = attribute
		self.split_value = split_value
		self.left = left
		self.right = right

'''
Takes in data from filepath
Returns 2D array of data point vectors
'''
def load_data(filepath):
	data_points = []
	with open(filepath) as file:
		for line in file.readlines():
			data_points.append(line.strip().split(','))
	return np.array(data_points)

'''
Calculates the standard deviation across one attribute for a set of data points
'''
def stdev(data_subset, target):
	if len(data_subset) != 0:
		values = data_subset[:,target]
		return np.std([float(val) for val in values if val != '?'])
	else:
		 return 0

'''
Calculates the standard deviation of two subsets of data split on a single attribute
'''
def split_stdev(data_subset1, data_subset2, target):
	size = len(data_subset1) + len(data_subset2)
	return (len(data_subset1)/size) * stdev(data_subset1, target) + \
	(len(data_subset2)/size) * stdev(data_subset2, target)

'''
Breaks data set into two subsets based on a threshold value for a single attribute
'''
def split_data(data_set, target, attribute, split_point):
	subset1, subset2 = [], []
	for city in data_set:
		if city[attribute] == '?':
			r.choice([subset1, subset2]).append(city)
		elif float(city[attribute]) <= split_point:
			subset1.append(city)
		else:
			subset2.append(city)
	subset1, subset2 = np.array(subset1), np.array(subset2)
	reduction = stdev(data_set, target) - split_stdev(subset1, subset2, target)
	return reduction, subset1, subset2

'''
Finds best split for data set based on which threshold value for a given attribute
most reduces standard deviation
'''
def find_split(data_set, target, attribute):
	max_std_reduction = 0
	subset1 = []
	subset2 = []
	split_point = 0
	for split in np.arange(0.1, 1.0, 0.1):
		cur_reduction, cur_subset1, cur_subset2 = split_data(data_set, target, attribute, split)
		if cur_reduction > max_std_reduction :
			max_std_reduction = cur_reduction
			subset1 = cur_subset1
			subset2 = cur_subset2
			split_point = split

	return max_std_reduction, subset1, subset2, split_point

'''
Returns decision tree constructed using the ID3 algorithm
Recursively finds optimal attribute to split the data set or subset on and
branches there, continuing until depth is satisfied or data is fully divided
'''
def id3(data_set, target, depth):
	if depth == 0:
		return [float(data_point[target]) for data_point in data_set]
	else:
		max_attribute = 0
		max_reduction = 0
		subset1 = []
		subset2 = []
		split_value = 0
		for i in set(range(len(data_set[0]))) - set([target, 0, 1, 2, 3]):
			reduction, temp_subset1, temp_subset2, split = find_split(data_set, target, i)
			if reduction > max_reduction:
				max_reduction = reduction
				max_attribute = i
				subset1 = temp_subset1
				subset2 = temp_subset2
				split_value = split
		if len(subset1) == 0:
			return [float(data_point[target]) for data_point in data_set]
		elif len(subset2) == 0:
			return [float(data_point[target]) for data_point in data_set]
		left = id3(subset1, target, depth - 1)
		right = id3(subset2, target, depth - 1)
		node = Node(max_attribute, split_value, left, right)
		return node

'''
Recursively traverses decision tree to classify target attribute for new data point
'''
def classify(data_point, decision_tree):
	if type(decision_tree) == list:
		return np.mean(decision_tree)
	elif data_point[decision_tree.attribute] == '?':
		return classify(data_point, r.choice(decision_tree.left, decision_tree.right))
	elif float(data_point[decision_tree.attribute]) <= decision_tree.split_value:
		return classify(data_point, decision_tree.left)
	else:
		return classify(data_point, decision_tree.right)

def main():
	data = np.array([[.1, 0, .1, .1],
					[.2, 0, .1, .2],
					[.3, 0, '?', .2],
					[.4, 0, .2, .4],
					[.5, 0, .3, .5]])

	data_point = [.4, 0, .2, .4]

	tree = id3(data, 0, 2)
	print(classify(data_point, tree))

if __name__ == '__main__':
	main()
