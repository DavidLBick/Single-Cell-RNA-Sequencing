import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
np.random.seed(0)
import palettable


# usage: plot_pca.py <h5_data_path> <n_rows to include>

def load_data(h5_path, n_rows='all'):
	store = pd.HDFStore(h5_path)
	if n_rows = 'all':
		feature_matrix_dataframe = store['rpkm']
		labels = store['labels']
	else:
		feature_matrix_dataframe = store['rpkm'].iloc[:n_rows]
		labels = store['labels'].iloc[:n_rows]
	unique_labels = list(np.unique(labels))
	store.close()
	return feature_matrix_dataframe, labels, unique_labels

def do_pca(data, n_components=2):
	pca = PCA(n_components=n_components)
	transformed_data = pca.fit_transform(data)
	var_per_component = pca.explained_variance_ 
	return transformed_data, var_per_component

def plot_data(data, labels, unique_labels):
	# plot 2D data
    fig, ax = plt.subplots(1)
	ax.set_color_cycle(palettable.mycarta.Cube1_16.mpl_colors)
		
	for label in unique_labels:
		data_with_this_label = data[labels==label]
		ax.scatter(data_with_this_label[:,0],data_with_this_label[:,1],label=label)
		
	fig.savefig('pca_plot.pdf')
	print('data plotted and saved in pca_plot.pdf')

def main(data_path, n_rows='all'):
	feats, labels, unique_labels = load_data(data_path, n_rows)
	transformed_data = do_pca(feats)
	plot_data(transformed_data, labels, unique_labels)

if __name__ == '__main__':
	if len(sys.argv < 2):
		print('no data path given')
	elif len(sys.argv == 2):
		main(str(sys.argv[1]))
    else: 
		main(str(sys.argv[1]), int(sys.argv[2]))
	
	
	