import magic
import sys
from data import Data
import numpy as np

def main(data_path, n_rows):
	data = Data(data_path, n_rows)
	data.load_data()
	magic_operator = magic.MAGIC()
	X_magic = magic_operator.fit_transform(data.dataframe)
	filename = 'magic_data_{}_rows.npy'
	output_file = 'magic_data_from_{}_{}_rows.npy'.format(data_path.replace('.', '').replace('/', ''), n_rows)
	np.save(output_file, X_magic)
	print('data saved in', output_file)
	
if __name__ == '__main__':
	if len(sys.argv) < 2:
		print('provide data path and optional n_rows')
	elif len(sys.argv) == 2:
		main(sys.argv[1], n_rows='all')
	elif len(sys.argv) == 3:
		main(sys.argv[1], n_rows=int(sys.argv[2]))
