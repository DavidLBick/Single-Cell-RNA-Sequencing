class Data:

	def __init__(h5_path, n_rows='all'):
		self.path = h5_path
		self.n_rows = n_rows
		self.features = None
		self.labels = None
		self.unique_labels = None
		self.relabels = None
		
	def load_data():
		store = pd.HDFStore(self.path)
		if self.n_rows == 'all':
			feature_matrix_dataframe = store['rpkm']
			labels = store['labels']
		else:
			feature_matrix_dataframe = store['rpkm'].iloc[:self.n_rows]
			labels = store['labels'].iloc[:self.n_rows]
		unique_labels = list(np.unique(labels))
		store.close()
		self.features = feature_matrix_dataframe
		self.labels = labels
		self.unique_labels = unique_labels
		return self.features, self.labels, self.unique_labels
	
	def relabel(label):
		self.relabels = self.labels==label
		
	