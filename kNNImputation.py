import pandas as pd
from missingpy import KNNImputer

class KNN_impute:
        
    def __init__(self,missing_val_rep=0.0, k = 10, copy = False):
        
        self.missing_val_rep = missing_val_rep
        self.imputer = KNNImputer(missing_val_rep,k,copy=copy)
    
    def fit(self,X,y):
        
        def add_median(df):
            medians = df.median(axis=0)
            return df.replace(self.missing_val_rep, medians.to_dict())
        
        X['labels'] = y
        X_median = X.groupby(by='labels').apply(add_median)
        print(X_median)
        X.drop(columns=['labels'],inplace=True)
        X_median.drop(columns=['labels'],inplace=True)

        self.imputer.fit(X_median)
        
    
    def transform(self,X):
        return self.imputer.transform(X)


if __name__ == '__main__':
    
    X = pd.DataFrame([ [0,100,1],
                       [5,120,1],
                       [10,90,1],
                       [0,110,1],
                       [0,100,1],
                       [100,120,1],
                       [50,90,1],
                       [0,110,1]])
    
    y = pd.Series([1,1,1,1,0,0,0,0])
    
    obj = KNN_impute(0,2,copy=True)
    obj.fit(X,y)
    #print(X)
    #print(obj)
    print(obj.transform(X))
    