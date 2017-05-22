
'''
Aidan Holloway-Bidwell and Allie Warren
Data Mining
Final Project
Decision Trees
'''

import numpy as np

def load_data(filepath):
	data_points = []
	with open(filepath) as file:
		for line in file.readlines():
			data_points.append(line.strip().split(','))
	return np.array(data_points)

def stdev(data_subset, target):
	values = data_subset[:,target]
	return np.std([float(val) for val in values if val != '?'])

def split_stdev(data_subset1, data_subset2, target):
	size = len(data_subset1) + len(data_subset2)
	return (len(data_subset1)/size) * stdev(data_subset1, target) + \
	(len(data_subset2)/size) * stdev(data_subset2, target)

def split_data(data_set, target, attribute, split_point):
	subset1 = np.array([city in data_set if city[attribute] <= split_point])
	subset2 = np.array([city in data_set if city[attribute] > split_point])
	reduction = stdev(data_set, target) - split_stdev(subse1, subset2, target)
	return reduction, subset1, subset2

def find_split(data_set, target, attribute):
	max_std_reduction = 0
	subset1 = []
	subset2 = []
	for split in range(.1, 1, .1):
		cur_reduction, cur_subset1, cur_subset2 = split_data(data_set, target, attribute, split)
		if(cur_reduction <= max_std_reduction):
			max_std_reduction = cur_reduction
			subset1 = cur_subset1
			subset2 = cur_subset2

	return max_std_reduction, subset1, subset2




def main():
	print(std_dev(np.array([[1],[2],[3],[4],['?']]), 0))

if __name__ == '__main__':
	main()
