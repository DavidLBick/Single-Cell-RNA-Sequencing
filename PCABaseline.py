import sys
from Evaluation import EvalClustering
from dataloader import Loader
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def main(data_path,out_path,n_components):
    
    # Can change this initialization based on how data loading occurs
    X_train, y_train = Loader.get_train()
    X_test, y_test = Loader.get_test()
    
    # Fit model
    model = PCA( n_components = n_components )
    model.fit(X_train)
    
    # Evaluate model
    X_test_transformed = model.transform(X_test)
    res = EvalClustering.evaluate(KMeans,X_test_transformed,y_test,iters=10,n_clusters=n_components)
    
    # Write to the output
    res.to_csv(out_path)

if __name__ == '__main__':
    
    data_path,out_path,n_component = sys.argv[1:4]
    
    main(data_path,out_path,n_component)