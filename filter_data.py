import numpy as np
from os.path import basename
import sys

def get_file_root_name(file_name):
	return basename(file_name).split('.')[0]

if __name__ == '__main__':
	print('first argument: data path, second argument: indices')
	npy_data_path = str(sys.argv[1])
	data_name = get_file_root_name(npy_data_path)
	npy_col_idxs = str(sys.argv[2])
	idxs = np.load(npy_col_idxs)
	idxs_name = get_file_root_name(npy_col_idxs)
	file_out = data_name + '_filtered_with_' + idxs_name + '.npy'
	out_data = np.load(npy_data_path)[:, idxs]
	np.save(file_out, out_data)
	print('out size:', out_data.shape)	
	print('file filtered and saved in', file_out)
	
