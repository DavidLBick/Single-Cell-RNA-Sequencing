import sys
import keras
from keras.models import Sequential, Model, load_model, save
from keras.layers import Dense, Activation
from Data import data
import numpy as np


def load_data(feat_path, label_path, n_rows):
	data = Data(label_path, n_rows, feat_path)
	data.load_data()
	return data.imputed_features, data.one_hot, data.n_classes 

	
def create_model(input_sze, n_classes):
	model = Sequential([
		Dense(1000, input_shape=(input_sze,)),
		Activation('relu'),
		Dense(n_classes,  name='output_layer')
	])

	model.compile(optimizer='adam',
				  loss='categorical_crossentropy',
				  metrics=['accuracy', 'categorical_accuracy'])
	
	return model
	

def train_model(model, data, labels):
	model.fit(data, labels, epochs=10, batch_size=32)
	filename = 'trained_model.h5'
	model.save('trained_model.h5')
	return filename

	
def test_model(model_file, data):
	model = load_model(model_file)
	layer_name = 'output_layer'
	intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
	embeddings = intermediate_layer_model.predict(data).to_numpy()
	np.save('test_embeddings.npy', embeddings)
	
		
def main(feat_path, label_path, n_rows):
	feats, labels, n_classes = load_data(feat_path, label_path, n_rows)
	model = create_model(input_sze=feats.shape[1], n_classes=n_classes)
	model_file = train_model(model)
	test_model(model_file)

if __name__ == '__main__':
	main(sys.argv[1], sys.argv[2], sys.argv[3])
	
