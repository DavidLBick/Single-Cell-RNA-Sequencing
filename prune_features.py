import numpy as np
import math
from omp_feature_selection import n_unique

def n_features_not_close_enough(n_classes, current_amount, target_n):
	if math.abs(current_amount - target_n) > n_classes:
		return True
	return False
	

def remove_some(n_classes, current_amount, target_n, selections, new_set):
	pass
	
	
def add_some(n_classes, current_amount, target_n, selections, new_set):
	pass
	
	
def prune_to_correct_amount(selections, target_n):
	n_classes = selections.shape[0]
	current_amount = n_unique(selections)
	current_cutoff_rank = selections.shape[0]
	new_set = selections
	while n_features_not_close_enough(n_classes, current_amount, target_n):
		if current_amount > target_n:
			new_set, current_amount = remove_some(n_classes, current_amount, target_n, selections, new_set)
		elif current_amount < target_n:
			new_set, current_amount = add_some(n_classes, current_amount, target_n, selections, new_set)
	return new_set
	
		
def simple_prune_to_correct_amount(selections, target_n):
	print("pruning features...")
	current_amount = n_unique(selections)
	while current_amount > target_n:
		selections = selections[:, :-1]  # remove at max <n_classes> features at a time
		current_amount = n_unique(selections)
	return selections

	
def main(data_path, n_features):
	pass 

if __name__ == '__main__':
	print('not implemented')
