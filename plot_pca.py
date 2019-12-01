import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
np.random.seed(0)
import palettable
import matplotlib.pyplot as plt
from data import Data

# usage: plot_pca.py <h5_data_path> <n_rows to include (optional)> <whiten data (bool) (optional,  default 0)>

def do_pca(data, whiten, n_components=2):
	pca = PCA(n_components=n_components, whiten=whiten)
	transformed_data = pca.fit_transform(data)
	var_per_component = pca.explained_variance_ratio_  
	print("fraction of variance explained per chosen component:", var_per_component)
	return transformed_data, var_per_component

def plot_data(data, labels, unique_labels, n_rows, whiten):
	# plot 2D data
	# print("unique labels:", unique_labels)
	# print('labels:', labels)
	fig, ax = plt.subplots(1)
	if not whiten:
		ax.set_xscale('symlog')
		ax.set_yscale('symlog')
	# ax.set_prop_cycle('color', palettable.mycarta.Cube1_16.mpl_colors)
	for label in unique_labels:
		data_with_this_label = data[labels==label]
		# print("label", label)
		# print("data_with_this_label", labels==label)
		ax.plot(data_with_this_label[:,0],data_with_this_label[:,1],label=label, marker='.', linestyle='None')
		
	
	handles, labels = ax.get_legend_handles_labels()
	lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,-0.1))
	fig_name = 'pca_plot_{}_rows_whiten_{}.pdf'.format(n_rows, whiten)
	fig.savefig(fig_name, bbox_extra_artists=(lgd,), bbox_inches='tight')
	print('data plotted and saved in', fig_name)

def main(data_path, n_rows, whiten):
	data = Data(data_path, n_rows)
	feats, labels, unique_labels = data.load_data()
	transformed_data, _ = do_pca(feats, whiten)
	plot_data(transformed_data, labels, unique_labels, n_rows, whiten)

if __name__ == '__main__':
	if len(sys.argv) < 2:
		print('no data path given')
	elif len(sys.argv) == 2:
		main(str(sys.argv[1]), n_rows='all', whiten=False)
	elif len(sys.argv) == 3:
		main(str(sys.argv[1]), n_rows=int(sys.argv[2]), whiten=False)
	elif len(sys.argv) == 4:
		main(str(sys.argv[1]), n_rows=int(sys.argv[2]), whiten=bool(sys.argv[3]))
		