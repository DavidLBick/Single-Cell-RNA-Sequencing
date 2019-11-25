from log_omp import LogisticOMP
from data import Data
import sys
import math
import numpy as np
# do feature selection for each class separately and take union of the selected features
# usage: python omp_feature_selection.py <data_path> <n_features_to_select> <n_rows (optional)>

def n_unique(array):
	return np.unique(array).size
	
	
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
	current_amount = np.unique(selections)
	current_cutoff_rank = selections.shape[0]
	new_set = selections
	while n_features_not_close_enough(n_classes, current_amount, target_n):
		if current_amount > target_n:
			new_set, current_amount = remove_some(n_classes, current_amount, target_n, selections, new_set)
		elif current_amount < target_n:
			new_set, current_amount = add_some(n_classes, current_amount, target_n, selections, new_set)
	return new_set
	
		
def simple_prune_to_correct_amount(selections, target_n):
	current_amount = np.unique(selections)
	while current_amount > target_n:
		selections = selections[:, :-1]  # remove at max <n_classes> features at a time
		current_amount = np.unique(selections)
	return selections	


def select_features(data, n_features):
	all_selections = np.array((data.shape[0], n_features))
	for i, label in enumerate(data.unique_labels):
		omp_selector = LogisticOMP(n_nonzero_coefs=n_features, eps=0.001)
		data.relabel(label)
		omp_selector.fit(data.features, data.relabels)
		feat_idx, ranked_features = omp_selector.get_selected_feature_idxs()
		all_selections[i] = ranked_features
		print("ranked features selected for {}: {}".format(label, ranking))
		print("ranked features selected for {} found".format(label))
	
	print("pruning features...")
	final_set = simple_prune_to_correct_amount(all_selections, n_features)
	print("size of final set:", final_set.shape)
	print("final set:", final_set)
	return final_set


def main(data_path, n_features, n_rows):
	# print(data_path)
	# print(n_rows)
	data = Data(h5_path=data_path, n_rows=n_rows)
	data.load_data()
	selected_set = select_features(data, n_features)
	output_file = '{}_selected_features_from_{}_{}_rows.npy'.format(n_features, data_path, n_rows)
	np.save(output_file, selected_set)
	print('selected set saved in', output_file)

if __name__ == '__main__':
	if len(sys.argv) < 3:
		print('no data path or number of features given')
	elif len(sys.argv) == 3:
		main(str(sys.argv[1]), n_features=int(sys.argv[2]), n_rows='all')
	elif len(sys.argv) == 4:
		# print('3 arguments given')
		main(data_path=str(sys.argv[1]).strip(), n_features=int(sys.argv[2]), n_rows=int(sys.argv[3]))