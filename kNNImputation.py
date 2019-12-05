import pandas as pd
from missingpy import KNNImputer
import numpy as np
import config
from tqdm import tqdm

class KNN_impute:
        
    def __init__(self,missing_val_rep=0.0, k = 10, copy = False):
        
        self.missing_val_rep = missing_val_rep
        self.imputer = KNNImputer(missing_val_rep,k,copy=copy,col_max_missing=1.0, row_max_missing=01.0)

    
    def add_medians(self,X,y):
        
        X['labels'] = y
        label_meds = remove_rows(X).groupby(by='labels').median()
        #print(label_meds)
        for l in tqdm(label_meds.index):
            X[X['labels']==l] = X[X['labels']==l].replace(self.missing_val_rep,label_meds.loc[l,:].to_dict())
        
        X.drop(columns=['labels'],inplace=True)
    
    def fit(self,X,y):
        
        self.add_medians(X,y)
        print('INSIDE IMPUTER: Beginning the fit')
        self.imputer.fit(X)        
        print('INSIDE IMPUTER: Completed the fit')
        return None
        
        '''
        def add_median(df):
            medians = df.median(axis=0)
            return df.replace(self.missing_val_rep, medians.to_dict())
        
        X['labels'] = y
        X_median = X.groupby(by='labels').apply(add_median)
        #print(X_median)
        X.drop(columns=['labels'],inplace=True)
        X_median.drop(columns=['labels'],inplace=True)

        self.imputer.fit(X_median)
        '''
    
    def transform(self,X):
        return self.imputer.transform(X)

def remove_rows(X,y=None):
    to_drop = X.index[(X==0).all(axis=1)]         
    
    if y is None:
        return X.drop(index=to_drop)
        
    return X.drop(index=to_drop), y.drop(to_drop)

def load_df(path):
    store = pd.HDFStore(path)
    X, y = store['rpkm'], store['labels']
    
    return X,y

def join_sets(X1,y1,X2,y2):
    X, y = pd.concat([X1, X2]), pd.concat([y1, y2]) 
    #return X.reset_index().drop(columns = ['index']), y.reset_index().iloc[:,1]
    return X, y
    
def split_sets(X,y,l1):
    return X.iloc[:l1,:], y.iloc[:l1], X.iloc[l1:,:], y.iloc[l1:]
    
def run_imputation(train_path, test_path=None):
    
    name = '../data/kNNImputedTrain.npy'
    name_y = '../data/kNNImputedTrain_y.npy'
    X, y = remove_rows(*load_df(train_path))
    print('Data Loaded')
    
    if test_path is not None:
        l1 = len(X)
        X, y = join_sets(X,y,remove_rows(*load_df(test_path)))
        name = '../data/kNNImputedTest.npy'
        name_y = '../data/kNNImputedTest_y.npy'
    
    imputer = KNN_impute()
    imputer.fit(X,y)
    print('Completed the imputer fit')
    
    X_transformed = imputer.transform(X)
    y_transformed = y
    
    print('Completed Transformation')
    
    if test_path is not None:
        _, _, X_transformed, y_transformed  = split_sets(X,y,l1)
    
    print('Beginning to save')
    np.save(name, X_transformed)
    np.save(name_y, y_transformed)
    print('Completed saving')
    
def run_imputation_direct():
    
    train_path = config.TRAIN_DATA_PATH
    test_path = config.TEST_DATA_PATH

    print('Running imputation for train')    
    run_imputation(train_path)
    print('Running imputation for test')
    run_imputation(train_path,test_path)    

def test_median_calculation():
    
    train_path = config.TRAIN_DATA_PATH
    X, y = load_df(train_path)
    print('Data Loaded')
    
    X, y = remove_rows(X,y)
    X['labels'] = y
    
    print(X.groupby(by='labels').median())
    

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
    obj.add_medians(X,y)
    #print(X)
    #print(obj)
    #print(obj.transform(X))
    print(X)
    
    #run_imputation('blah','blah')
    