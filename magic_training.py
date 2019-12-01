import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from Data import data

def load_data(feat_path, label_path, n_rows):
	data = Data(label_path, n_rows, feat_path)
	data.load_data()
	return data.imputed_features, data.labels

	
def create_model(input_sze):
	model = Sequential([
		Dense(32, input_shape=(input_sze,)),
		Activation('relu'),
		Dense(10)
	])

	model.compile(optimizer='adam',
				  loss='categorical_crossentropy',
				  metrics=['accuracy', 'categorical_accuracy'])
	
	return model
	

def train_model(model, data):
	model.fit(data, labels, epochs=10, batch_size=32)

# save model