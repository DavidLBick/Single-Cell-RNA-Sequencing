import pandas as pd
import numpy as np

class Data:
	def __init__(self, h5_path, n_rows='all', imputed_feature_path=None):
		self.path = h5_path
		self.n_rows = n_rows
		self.features = None
		self.labels = None
		self.unique_labels = None
		self.relabels = None
		self.dataframe = None
		self.imputed_features = None
		if imputed_feature_path is not None:
			self.imputed_feature_path = imputed_feature_path
			self.has_imputed_features = True
		else:
			self.has_imputed_features = False

	def load_data(self):
		store = pd.HDFStore(self.path)
		if self.n_rows == 'all':
			feature_matrix_dataframe = store['rpkm']
			labels = store['labels']
			if self.has_imputed_features:
				self.imputed_features = np.load(self.imputed_feature_path)
		else:
			feature_matrix_dataframe = store['rpkm'].iloc[:self.n_rows]
			labels = store['labels'].iloc[:self.n_rows]
			if self.has_imputed_features:
				self.imputed_features = np.load(self.imputed_feature_path)[:self.n_rows]
		unique_labels = list(np.unique(labels))
		store.close()
		self.dataframe = feature_matrix_dataframe
		self.features = feature_matrix_dataframe.to_numpy()
		self.labels = labels.to_numpy()
		self.unique_labels = unique_labels
		print("{} rows data loaded".format(self.n_rows))
		return self.features, self.labels, self.unique_labels
	
	def relabel(self, label):
		self.relabels = self.labels==label
		
	
