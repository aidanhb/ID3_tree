
'''
Aidan Holloway-Bidwell and Allie Warren
Data Mining
Final Project
Decision Trees
'''

import numpy as np
import random as r
from matplotlib import pyplot as plt

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
def load_data(filepath, limit=None):
	data_points = []
	with open(filepath) as file:
		to_load = file.readlines()
		if not limit:
			limit = len(to_load)
		for line in to_load[:limit]:
			data_points.append(line.strip().split(','))
	return np.array(data_points)

def load_attributes(filepath):
	with open(filepath) as file:
		attributes = [line.strip() for line in file.readlines()]
	return attributes

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
def id3(data_set, target, depth, attributes, repeats=True):
	if depth == 0:
		return [float(data_point[target]) for data_point in data_set if data_point[target] != '?']
	else:
		max_attribute = 0
		max_reduction = 0
		subset1 = []
		subset2 = []
		split_value = 0
		for i in attributes:
			reduction, temp_subset1, temp_subset2, split = find_split(data_set, target, i)
			if reduction > max_reduction:
				max_reduction = reduction
				max_attribute = i
				subset1 = temp_subset1
				subset2 = temp_subset2
				split_value = split
		if not repeats:
			attributes = attributes - set([max_attribute])
		if len(subset1) == 0:
			return [float(data_point[target]) for data_point in data_set if data_point[target] != '?']
		elif len(subset2) == 0:
			return [float(data_point[target]) for data_point in data_set if data_point[target] != '?']
		left = id3(subset1, target, depth - 1, attributes, repeats)
		right = id3(subset2, target, depth - 1, attributes, repeats)
		node = Node(max_attribute, split_value, left, right)
		return node

'''
Recursively traverses decision tree to classify target attribute for new data point
'''
def classify(data_point, decision_tree):
	if type(decision_tree) == list:
		return np.mean(decision_tree)
	elif data_point[decision_tree.attribute] == '?':
		return classify(data_point, r.choice([decision_tree.left, decision_tree.right]))
	elif float(data_point[decision_tree.attribute]) <= decision_tree.split_value:
		return classify(data_point, decision_tree.left)
	else:
		return classify(data_point, decision_tree.right)

def classify_test_data(test_data, decision_tree):
	classifications = []
	for city in test_data:
		classifications.append(classify(city, decision_tree))

	return classifications

def traverse(tree, f, attributes, ID):
	if type(tree) == list:
		return
	else:
		left_random, right_random = r.randint(1,10000), r.randint(1,10000)
		if type(tree.left) == list:
			f.write('\"{0} split on {1} ({3})\" -> \"{2}\";\n'.format(attributes[int(tree.attribute)], \
																tree.split_value, \
																np.mean(tree.left),
																ID))
		else:
			f.write('\"{0} split on {1} ({4})\" -> \"{2} split on {3} ({5})\";\n'.format(attributes[int(tree.attribute)], \
																			tree.split_value, \
																			attributes[int(tree.left.attribute)], \
																			tree.left.split_value,
																			ID,
																			left_random))
		if type(tree.right) == list:
			f.write('\"{0} split on {1} ({3})\" -> \"{2}\";\n'.format(attributes[int(tree.attribute)], \
																tree.split_value, \
																np.mean(tree.right),
																ID))
		else:
			f.write('\"{0} split on {1} ({4})\" -> \"{2} split on {3} ({5})\";\n'.format(attributes[int(tree.attribute)], \
																			tree.split_value, \
																			attributes[int(tree.right.attribute)], \
																			tree.right.split_value,
																			ID,
																			right_random))

		traverse(tree.left, f, attributes, left_random)
		traverse(tree.right, f, attributes, right_random)



def print_tree(tree, filename, attributes):
	with open(filename, 'w') as f:
		f.write("digraph tree{\n")
		traverse(tree, f, attributes, 0)
		f.write("}\n")

def sse(test_set, predicted_values, target):
	sum_squared_error = 0
	count = 0
	for i in  range(len(test_set)):
		if test_set[i, target] != "?":
			sum_squared_error = sum_squared_error + (float(test_set[i, target])-float(predicted_values[i]))**2
			count = count + 1

	return sum_squared_error, count

def create_train_test(data):
	r.shuffle(data)
	mid = int(len(data)/2)
	train = data[:mid]
	test = data[mid:]

	return train, test



def main():
	sses = []
	for size in [100,200,500,1000,1500]:
		print('Doing {0}'.format(size))
		data = load_data('communities.data', size)
		attributes = load_attributes('attributes.txt')
		training_data, test_data = create_train_test(data)
		# tree_data = data[1:]
		targets = [17, 90, 95, 109, 122, 127]
		target = targets[1]
		attributes_to_use = set(range(len(data[0]))) - set([target, 0, 1, 2, 3, 4])
		tree = id3(training_data, target, 10, attributes_to_use, False)
		# data_point = data[0]
		# print(classify(data_point, tree))
		print_tree(tree, "tree_file.dot", attributes)
		classifications = classify_test_data(test_data, tree)
		tup = sse(test_data, classifications, target)
		sses.append(tup[0]/tup[1])
	print(sses)
	plt.plot(sses)
	plt.show()

if __name__ == '__main__':
	main()
