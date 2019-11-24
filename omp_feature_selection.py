from log_omp import LogisticOMP
from data import Data
import sys

# do feature selection for each class separately and take union of the selected features
# usage: omp_feature_selection <data_path> <n_rows (optional)>

def select_features(data):
	for label in data.unique_labels:
		omp_selector = LogisticOMP(n_nonzero_coefs=5, eps=0.001)
		data.relabel(label)
		omp_selector.fit(data.features, data.relabels)
		feat_idx, ranking = omp_selector.get_selected_feature_idxs()
		print("features selected for {}: {}".format(label, feat_idx))		

def main(data_path, n_rows):
	# print(data_path)
	# print(n_rows)
	data = Data(h5_path=data_path, n_rows=n_rows)
	data.load_data()
	select_features(data)

if __name__ == '__main__':
	if len(sys.argv) < 2:
		print('no data path given')
	elif len(sys.argv) == 2:
		main(str(sys.argv[1]), n_rows='all')
	elif len(sys.argv) == 3:
		# print('3 arguments given')
		main(data_path=str(sys.argv[1]).strip(), n_rows=int(sys.argv[2]))