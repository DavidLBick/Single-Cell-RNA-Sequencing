# Set constants, magic numbers, variables, settings, etc. here
BATCH_SIZE = 64
MODELS_PATH = './saved_model/'
SAVE_MODEL_NAME = "baseline_model"
N_EPOCHS = 3
TRAIN_DATA_NP_ARRAY = 'magic_data_from_train_datah5_all_rows_filtered_with_1013_selected_features_from_train_datah5_all_rows.npy' 
TEST_DATA_NP_ARRAY = 'magic_imputed_test_set_filtered_with_1013_selected_features_from_train_datah5_all_rows.npy' 
DATA_PATH = '../'
TRAIN_DATA_PATH = DATA_PATH + 'training_data.h5'
TEST_DATA_PATH = DATA_PATH + 'testing_data.h5'
TRAIN_FLAG = True
TEST_FLAG = True
N_CLASSES = 46
INPUT_SIZE = 20499
EMBEDDINGS_OUTPUT_FILE = 'final_embeddings_5_1000_layers_raw_data.npy'
LABELS_OUTPUT_FILE = 'final_labels_5_1000_layers_raw_data.npy'
