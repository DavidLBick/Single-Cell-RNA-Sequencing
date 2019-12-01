import magic
import sys
import numpy as np
import pandas as pd


def concat(train_path, test_path):
	train_store = pd.HDFStore(train_path)
	test_store = pd.HDFStore(test_path)
	train_features = train_store['rpkm']
	train_len = len(train_features.index)
	test_features = test_store['rpkm']
	combined_features = pd.concat([train_features, test_features])
	return combined_features, train_len

	
def do_magic(dataframe):
	magic_operator = magic.MAGIC()
	magic_set = magic_operator.fit_transform(dataframe)
	return magic_set
	
	
def main(train_path, test_path):
	combined_features, train_len = concat(train_path, test_path)
	imputed_set = do_magic(combined_features)
	imputed_test_set = imputed_set.iloc[train_len:, :].to_numpy()
	filename = 'magic_imputed_test_set.npy'
	np.save(filename, imputed_test_set)
	print('file saved in:', filename)
	
	
if __name__ == '__main__':
	main(sys.argv[1], sys.argv[2])
