"""
@author: Mariana Busarello, 2025
"""

import os
import geopandas as gpd
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import joblib
import argparse

prediction_path = '/app/Mariana/Project2/models/find_best/'
input_path = '/app/Mariana/Project2/dataset2/round2_new/sampled/'

model_path = '' #fullpath to the joblib model


def fitting(input_path,prediction_path,model_path):
    for inf_shapefile in os.listdir(input_path):
        if inf_shapefile.endswith('.shp'):
            print(inf_shapefile)
            inf_shp_path = os.path.join(input_path,inf_shapefile)
            fid_shp = gpd.read_file(inf_shp_path)
            fid_shp['FID'] = fid_shp.index
            inf_read = gpd.read_file(inf_shp_path)
            inf_read['FID'] = inf_read.index
            inf_data = inf_read.drop(columns=['geometry']).reset_index(drop=True)
            inf_data = inf_data.dropna()
    
            X_pred = inf_data[list(inf_data.drop('id',axis=1).columns)]
        
            X_pred = X_pred[['afs_min', 'afs_max', 'afs_mean', 'afs_median', 'aul_max', 'mul_median', 'facc_max', 'uds_min', 'uds_max', 'uds_median', 'sinuosity','class_dl']]
            
            onehot = OneHotEncoder(sparse_output=False)
            encoded_unseen = onehot.transform(X_pred[['class_dl']])
            encoded_unseen_df = pd.DataFrame(encoded_unseen, columns=onehot.get_feature_names_out(['class_dl']))
            X_pred = pd.concat([X_pred, encoded_unseen_df], axis=1).drop('class_dl', axis=1)
            
            model = joblib.load(model_path)
            
            predictions = model.predict(X_pred)
            probability = model.predict_proba(X_pred)
    
            predicted_proba = pd.DataFrame(probability, index=X_pred.index, columns=[f'prob_{i}' for i in range(probability.shape[1])])
            predictions_df = pd.DataFrame(predictions, index=X_pred.index, columns=['predicted'])
    
            output_df = inf_read.join(predictions_df).join(predicted_proba, how='left')
            output_df['FID'] = fid_shp['FID']
    
            output_df.to_file(os.path.join(prediction_path,'prediction_'+inf_shapefile))
            
if __name__ == '__main__':   
    parser = argparse.ArgumentParser(description='uses a joblib XGBoost model to predict if a channel segment is a ditch or a stream',
                                     add_help=True,formatter_class=argparse.HelpFormatter)
    parser.add_argument('input_path', help='path of the buffer data to be used to predict the classes')
    parser.add_argument('prediction_path', help='destination path for the prediction (output is a shapefile with a "predicted" column')
    parser.add_argument('model_path', help='fullpath to the joblib file model')
    args = vars(parser.parse_args())
    fitting(**args) 