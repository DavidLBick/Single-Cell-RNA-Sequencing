import sklearn.metrics as metrics
import pandas as pd

class EvalClustering:
    
    
    @staticmethod
    def __fit_and_predict(algo,X,**algo_kwargs):
        preds = algo(**algo_kwargs).fit_transform(X)
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
        
        preds = EvalClustering.__fit_and_predict(algo,X,**algo_kwargs)
        
        return pd.Series({ k:v(y,preds) for k,v in eval_strategies.items() })
        
            