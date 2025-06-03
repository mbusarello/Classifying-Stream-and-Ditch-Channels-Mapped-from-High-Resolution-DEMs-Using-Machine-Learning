"""
@author: Mariana Busarello, 2025
"""

import xgboost as xgb
import os
import geopandas as gpd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import numpy as np
import joblib
from sklearn.utils.class_weight import compute_class_weight
import argparse

def train(buffer_path,output_path,split_size):

    
    n=0
    for shapefile in os.listdir(buffer_path):
        if shapefile.endswith('.shp'):
            print(shapefile)
            shp_ = os.path.join(buffer_path,shapefile)
            read_shp = gpd.read_file(shp_)
            read_shp = read_shp.drop('geometry',axis=1)
            if n == 0:
                data = read_shp
                data = data.reset_index(drop=True)
                n=n+1
            else:
                data = pd.concat([data,read_shp])
                data = data.reset_index(drop=True)
    
    
    X = data[list(data.drop('id',axis=1).columns)]
    
    
    X = X[['afs_min', 'afs_max', 'afs_mean', 'afs_median', 'aul_max', 
           'mul_median', 'facc_max', 'uds_min', 'uds_max', 'uds_median', 
           'sinuosity','class_dl']]
    
    X = X.dropna()
    
    onehot = OneHotEncoder(sparse_output=False)
    encoded = onehot.fit_transform(X[['class_dl']])
    encoded_df = pd.DataFrame(encoded, columns=onehot.get_feature_names_out(['class_dl']))
    X = pd.concat([X, encoded_df], axis=1).drop('class_dl', axis=1)
    #categories = onehot.categories_
    
    y = data['id']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    weights = {cls: weight for cls, weight in zip(classes, class_weights)}
    
    sample_weights = np.array([weights[label] for label in y_train])
    
    model = xgb.XGBClassifier(objective='multi:softmax', num_class=3, eval_metric='auc')
    model.fit(X_train, y_train, sample_weight=sample_weights)
    
    y_pred = model.predict(X_test)
    
    print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
    print("Recall: ", metrics.recall_score(y_test, y_pred,average=None))
    print("Precision: ", metrics.precision_score(y_test, y_pred,average=None))
    print("F1: ", metrics.f1_score(y_test, y_pred,average=None))
    
    labels = ['accuracy','recall','precision','f1']
    savedata = pd.DataFrame({labels[0]:metrics.accuracy_score(y_test, y_pred),
                             labels[1]:metrics.recall_score(y_test, y_pred,average=None),
                             labels[2]:metrics.precision_score(y_test, y_pred,average=None),
                             labels[3]:metrics.f1_score(y_test, y_pred,average=None)})
    
    savedata.to_csv(os.path.join(output_path,'eval_'+split_size+'m.csv'), sep=';', index=False)
    
    joblib.dump(model, os.path.join(output_path,'model_'+split_size+'m.joblib'))
    

if __name__ == '__main__':   
    parser = argparse.ArgumentParser(description='trains an XGBoost model to classify streams and ditches and stores it as a joblib file',
                                     add_help=True,formatter_class=argparse.HelpFormatter)
    parser.add_argument('buffer_path', help='path to buffers with the extracted training data')
    parser.add_argument('output_path', help='destination path for evaluation and the model')
    parser.add_argument('split_size', help='size of the split (for file-naming purposes)')
    args = vars(parser.parse_args())
    train(**args) 