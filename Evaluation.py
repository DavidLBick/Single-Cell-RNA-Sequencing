import sklearn.metrics as metrics
import pandas as pd
import sys
import dataloading
import numpy as np
from sklearn.cluster import KMeans

# usage:
# python Evaluation.py embeddings_path.npy save_results_path.csv

class EvalClustering:
    @staticmethod
    def __fit_and_predict(algo,X,**algo_kwargs):
        preds = algo(**algo_kwargs).fit_predict(X)
        return preds

    @staticmethod
    def evaluate(algo,X,y,iters=1,**algo_kwargs):

        '''
        Returns clustering based evaluation metrics

        algo: algorithm used for clustering
        X: dataset to cluster
        y: dataset labels
        iter: no.of times to repeat the experiment to average out the results
        **algo_kwargs: options for the algorithm

        Returns: returns a pandas series with all the relevant evaluation metrics
        '''

        eval_strategies = {'homogeneity_score':metrics.homogeneity_score,
                           'completeness_score':metrics.completeness_score,
                           'v_measure_score':metrics.v_measure_score,
                           'fowlkes_mallows_score':metrics.fowlkes_mallows_score,
                           'adjusted_rand_score':metrics.adjusted_rand_score,
                           'adjusted_mutual_info_score':metrics.adjusted_mutual_info_score,
                           }

        '''
        Not included other evaluation strategies as they are evoked slightly differently
        than the ones included above. Shouldn't be hard to do that but will wait till
        we get there.
        '''

        res = pd.Series({ k:0 for k,_ in eval_strategies.items() })

        for i in range(iters):
            preds = EvalClustering.__fit_and_predict(algo,X,**algo_kwargs)
            res += pd.Series({ k:v(y,preds) for k,v in eval_strategies.items() })

        return res/iters

class EvalClassification:
    @staticmethod
    def evaluate(model,X,y,labels=None):

        preds = model.predict(X)
        conf_mat = metrics.confusion_matrix(y,preds)

        if labels in None:
            labels = [ 'Class ' + str(i) for i in range(y.max()) ]

        return {'confusion matrix':pd.DataFrame(conf_mat,index=labels,columns=labels)}
    
    @staticmethod
    def evaluate_direct(y_true,y_preds,labels=None):

        conf_mat = metrics.confusion_matrix(y_true,y_preds)

        if labels in None:
            labels = [ 'Class ' + str(i) for i in range(y_true.max()) ]

        return {'confusion matrix':pd.DataFrame(conf_mat,index=labels,columns=labels)}

if __name__ == '__main__':
    import pdb
    pdb.set_trace()
    embeddings_path = sys.argv[2]
    out_path = sys.argv[3]
    n_components = 46
    embeddings = np.load(embeddings_path)
    true_labels = dataloading.test_dataset.labels
    res = EvalClustering.evaluate(KMeans,
                                  embeddings,
                                  true_labels,
                                  iters=10,
                                  n_clusters=n_components)
    # Write to the output
    res.to_csv(out_path)
