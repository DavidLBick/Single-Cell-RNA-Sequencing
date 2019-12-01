import sys
from Evaluation import EvalClustering
import dataloading
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pdb

# usage:
# python PCABaseline.py _.py embedding_comp.csv 100

def main(data_path, out_path, n_components):
    # Can change this initialization based on how data loading occurs
    train_dataset = dataloading.train_dataset
    X_train = train_dataset.features
    y_train = train_dataset.labels
    test_dataset = dataloading.test_dataset
    X_test = test_dataset.features
    y_test = test_dataset.labels

    label_idx_to_str = dataloading.label_idx_to_str
    label_str_to_idx = dataloading.label_str_to_idx

    # Fit model
    print("Fitting model...")
    model = PCA(n_components = n_components)
    model.fit(X_train)
    print("done")

    # Evaluate model
    print("Testing model...")
    X_test_embeddings = model.transform(X_test)
    # eval sklearn.KMeans embeddings using the true labels
    #pdb.set_trace()
    res = EvalClustering.evaluate(KMeans,
                                  X_test_embeddings,
                                  y_test,
                                  iters=10,
                                  n_clusters=n_components)
    print("done")

    # Write to the output
    res.to_csv(out_path)

if __name__ == '__main__':
    data_path, out_path, n_component = sys.argv[1:4]
    main(data_path, out_path, int(n_component))
