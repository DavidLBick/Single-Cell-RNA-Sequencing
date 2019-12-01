from log_omp import LogisticOMP
from data import Data
import sys
import math
import numpy as np
# do feature selection for each class separately and take union of the selected features
# usage: python omp_feature_selection.py <data_path> <n_features_to_select> <imputed_features npy filepath> <n_rows (optional)>
# python omp_feature_selection.py ../train_data.h5 10

def n_unique(array):
	return np.unique(array).size 
	
	
def select_features(data, target_n_features):
	n_classes = n_unique(data.unique_labels)
	n_features_per_class = int(target_n_features/n_classes)
	actual_n_features = int(n_features_per_class*n_classes)
	print('resulting number of features selected:', actual_n_features)
	
	if data.has_imputed_features:
		train_data = data.imputed_features
	else:
		train_data = data.features
	
	all_selections = np.zeros((data.features.shape[1]), dtype=bool)
	for i, label in enumerate(data.unique_labels):
		print("selecting features for class {}/{}".format(i, len(data.unique_labels)))
		omp_selector = LogisticOMP(n_nonzero_coefs=n_features_per_class, eps=0.001)
		data.relabel(label)
		omp_selector.fit(train_data, data.relabels, all_selections)
		new_selected_features = omp_selector.get_binary_selected_feature_vector()
		all_selections += new_selected_features
		print('total amount of features currently selected:', np.nonzero(all_selections)[0].size)
		temp_filename = "temp_features_{}_outof_{}_classes.npy".format(i, len(data.unique_labels))
		np.save(temp_filename, all_selections)
		print('temp features saved in:', temp_filename)
		
	idxs = np.nonzero(all_selections)[0]
	# print('idxs', idxs)
	
	# final_set = simple_prune_to_correct_amount(all_selections, n_features)
	# print("size of final set:", idxs.size)
	# print("number of unique features:", n_unique(all_selections))
	# print("final set:", final_set)
	return idxs

def main(data_path, n_features, n_rows, imputed_features):
	# print(data_path)
	# print(n_rows)
	data = Data(h5_path=data_path, n_rows=n_rows, imputed_feature_path=imputed_features)
	data.load_data()
	selected_set = select_features(data, n_features)
	output_file = '{}_selected_features_from_{}_{}_rows.npy'.format(n_features, data_path.replace('.', '').replace('/', ''), n_rows)
	np.save(output_file, selected_set)
	print('selected set saved in {}'.format(output_file))

if __name__ == '__main__':
	if len(sys.argv) < 3:
		print('no data path or number of features given')
	elif len(sys.argv) == 3:
		main(str(sys.argv[1]), n_features=int(sys.argv[2]), imputed_features=None, n_rows='all')
	elif len(sys.argv) == 4:
		# print('3 arguments given')
		main(data_path=str(sys.argv[1]).strip(), n_features=int(sys.argv[2]), imputed_features=str(sys.argv[3]), n_rows='all')
	elif len(sys.argv) == 5:
		main(data_path=str(sys.argv[1]).strip(), n_features=int(sys.argv[2]), imputed_features=str(sys.argv[3]), n_rows=int(sys.argv[4]))
