import sys
import numpy as np

def create_integer_labels(label_path, ordered_label_path):

	labels = np.load(label_path)
	ordered = np.load(ordered_label_path)
	integered_labels = []
	for label in labels:
		integer = np.where(ordered == label)
		integered_labels.append(integer)
	
	np.save('test_integer_labels.npy', np.array(integered_labels))
	
	print('integer labels created')
	
if __name__ == '__main__':
	create_integer_labels(sys.argv[1], sys.argv[2])
	
