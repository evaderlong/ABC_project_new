import os
import numpy as np
import pandas as pd
import json
import jsonpickle
import time
import logging
from tqdm import tqdm
import datetime

from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from train import RegressionModels

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# config = read_json(os.getcwd(), 'config.json')

def write_json(dct_path, dct_data):
    try:
        with open(dct_path, 'w') as fp:
            json.dump(dct_data, fp)
    except Exception as e:
        raise e
        logger.error(f"Could not write data to {dct_path}.")

def write_jsonpickle(dct_path, dct_data):
    try:
        with open(dct_path, 'w') as f:
            data = jsonpickle.encode(dct_data)
            f.write(data)
    except Exception as e:
        raise e
        logger.error(f"Could not write data to {dct_path}.")

def read_json(dct_path):
    try:
        with open(dct_path) as fp:
            input_dct = json.load(fp)
    except Exception as e:
        raise e
        logger.error("No data read for feature extraction/prediction.")
        input_dct = {}
    return input_dct

def read_jsonpickle(dct_path):
    try:
        with open(dct_path, 'r') as f:
            data = f.read()
            decoded_file = jsonpickle.decode(data)
            
    except Exception as e:
        logger.error("Could not read data from {}:{}".format(dct_path, e))
    return decoded_file

def parse_input(input_path, config):
    """join 2 datasets to give more features"""

    ABC_Hour_df = pd.read_csv(os.path.join(input_path, config['general']['input_file_hour'])).dropna()
    ABC_Minute_df = pd.read_csv(os.path.join(input_path, config['general']['input_file_minute'])).dropna()
    logger.info(f"hour data size: {len(ABC_Hour_df)}, minute data size {len(ABC_Minute_df)}")

    ABC_Hour_df['Date'] = ABC_Hour_df['Time_Hour'].astype('datetime64[ns]').dt.date
    ABC_Hour_df['Hour'] = ABC_Hour_df['Time_Hour'].astype('datetime64[ns]').dt.hour
    ABC_Hour_df['Time_Hour_clean'] = pd.to_datetime(ABC_Hour_df['Time_Hour'].apply(lambda x:x[:16]), format='%Y-%m-%d %H:%M')
    ABC_Hour_df = ABC_Hour_df.drop(columns=['Time_Hour'])

    ABC_Minute_df['Date'] =  pd.to_datetime(ABC_Minute_df['Time_Minute'], format='%d/%m/%Y %H:%M').dt.date
    ABC_Minute_df['Hour'] = pd.to_datetime(ABC_Minute_df['Time_Minute'], format='%d/%m/%Y %H:%M').dt.hour

    ABC_full_df = ABC_Minute_df.merge(ABC_Hour_df, how='left', on=['Date','Hour'])

    ABC_full_df['Time_Minute'] = pd.to_datetime(ABC_full_df['Time_Minute'], format='%d/%m/%Y %H:%M')
    ABC_full_df.index = ABC_full_df['Time_Minute']
    logger.info(f"full data size: {len(ABC_full_df)}")
    return ABC_full_df

def rename_cols(input_name, window):
    return input_name+'_'+window

def create_rolling_features(interval_df, input_features_df, rolling_windows, input_features_dict):
    
    mask_timeindex_15min = input_features_df.index.isin([idx for idx in input_features_df.index if idx.time()<=datetime.time(8,14,0)])
    mask_timeindex_1h = input_features_df.index.isin([idx for idx in input_features_df.index if idx.time()<=datetime.time(8,59,0)])
    
    input_features = list(input_features_dict.keys())
    """rolling featuers - TODO: automate the process with config file for rolling window"""
    for window in rolling_windows:

        if window =='15T':   #'15min'
            rolling_feature_df = input_features_df[input_features].rolling(window, closed='left').agg(input_features_dict)[~mask_timeindex_15min]
            rolling_feature_df = rolling_feature_df.append(input_features_df[input_features].rolling(15, closed='left').agg(input_features_dict)[mask_timeindex_15min])      
        if window =='1h':
            input_features_dict_copy = input_features_dict.copy()
            input_features_dict_copy.update({'Onehour_BidAsk_Spread_Perc':'mean'})
            input_features_copy = list(input_features_dict_copy.keys())
            rolling_feature_df = input_features_df[input_features_copy].rolling(window, closed='left').agg(input_features_dict_copy)[~mask_timeindex_1h]
            rolling_feature_df = rolling_feature_df.append(input_features_df[input_features_copy].rolling(60, closed='left').agg(input_features_dict)[mask_timeindex_1h]) 
        else:
            rolling_feature_df = input_features_df[input_features].rolling(window, closed='left').agg(input_features_dict)
            
        rolling_feature_df.columns = [rename_cols(col, window) for col in rolling_feature_df.columns] 
        
        interval_df = interval_df.merge(rolling_feature_df, left_index=True, right_index=True)
    
    """remove all records that won't have data for the maximum rolling window - TODO: automate the process with config file for rolling window """

    interval_df = interval_df.loc[interval_df.index>=min(interval_df.index)+datetime.timedelta(days=6)]
    interval_df = interval_df[~interval_df.isnull().any(axis=1)]
    logger.info(f"Rollinig features created, data size: {len(interval_df)}")
    return interval_df

def get_selling_cost(level_sizes, level_prices, volume):
    """Calculate the weighted average selling cost across levels given the selling volume"""
    level = 1
    unfulfilled_vol = volume
    vol_levels= []
    price_levels = []
    slippage = 0
 
    while unfulfilled_vol > 0:
#         print(f'current level:{level}')   
        
        if level <= len(level_sizes):
            size = level_sizes[level-1]
            
            if unfulfilled_vol > size:
                level_fulfilled_vol = size
                vol_levels.append((level,level_fulfilled_vol))
                price_levels.append((level, level_prices[level-1]))
                unfulfilled_vol = unfulfilled_vol-level_fulfilled_vol
#                 print(f'level unfulfilled:{level}')
#                 print(f'unfulfilled volume at this level:{unfulfilled_vol}')
                level+=1
#                 print(f'level unfulfilled, go to level:{level}')
                
            elif unfulfilled_vol<=size:
#                 print(f'level fulfilled:{level}')
                level_fulfilled_vol = unfulfilled_vol
                vol_levels.append((level,level_fulfilled_vol))
                price_levels.append((level, level_prices[level-1]))
                unfulfilled_vol = unfulfilled_vol-level_fulfilled_vol
                break
        else:
            slippage = unfulfilled_vol/volume
            break
    
    weighted_cost = (np.array([s[1] for s in vol_levels])*np.array([p[1] for p in price_levels])).sum()/np.array([s[1] for s in vol_levels]).sum()
    
    return weighted_cost, slippage
            
            
def get_cost_dataframe(row, bidsizes_input, bidprices_input, volume_input):
    level_bidsizes = row[bidsizes_input].values
    price_bidsizes = row[bidprices_input].values
    volume = row[volume_input]
    row['Selling_Cost_Label'], row['Slippage'] = get_selling_cost(level_bidsizes, price_bidsizes, volume)

    return row

def run_feature_extraction(input_df, create_labels, config):

    volset1_ls = config['fe']['vol_set1']
    volset2_ls = config['fe']['vol_set2']
    time_window = ''.join(config['fe']['time_window'])

    ABC_full_df = input_df

    Bidsizes_levels = [col for col in ABC_full_df.columns if 'BidSize' in col]
    Bidprices_levels = [col for col in ABC_full_df.columns if 'BidPrice' in col]

    """Create some features based on the data exploration analysis""" 
    ABC_full_df['Level1_Spread'] = ABC_full_df['L1_AskPrice'] - ABC_full_df['L1_BidPrice']
    ABC_full_df['Level1_Spread_Perc'] = (ABC_full_df['L1_AskPrice'] - ABC_full_df['L1_BidPrice']) / ABC_full_df['L1_AskPrice']
    ABC_full_df['Tot_BidSize'] = ABC_full_df.filter(regex='BidSize').sum(axis=1)
    ABC_full_df['Tot_AskSize'] = ABC_full_df.filter(regex='AskSize').sum(axis=1)
    ABC_full_df['L1_BidSize_perc'] = ABC_full_df['L1_BidSize'] / ABC_full_df['Tot_BidSize']
    ABC_full_df['L1_Asksize_perc'] = ABC_full_df['L1_AskSize'] / ABC_full_df['Tot_AskSize']
    ABC_full_df['Onehour_BidAsk_Spread_Perc'] = ABC_full_df['Avg_Bid_Ask_Spread']/ABC_full_df['VWAP'] 

    """Construct simulation data for a variety of selling volume within a fixed time window, currently set to 5 mins"""

    ABC_Level_Interval_Size = ABC_full_df.resample(time_window)[Bidsizes_levels].sum()
    ABC_Level_Interval_Price = ABC_full_df.resample(time_window)[Bidprices_levels].mean()

    ABC_Level_Interval = ABC_Level_Interval_Price.dropna().merge(ABC_Level_Interval_Size, how='left', left_index=True, right_index=True)

    volset1 = np.arange(volset1_ls[0],volset1_ls[1],volset1_ls[2])
    volset2 = np.arange(volset2_ls[0],volset2_ls[1],volset2_ls[2])
    volset_df = pd.DataFrame({'Vol':volset1.tolist() + volset2.tolist(),'Vol_Interval':(volset1/2).tolist() + (volset2/2).tolist()})
    
    ABC_Level_Interval['key'] = 0
    volset_df['key'] = 0
    ABC_Level_Interval['Time_Interval'] = ABC_Level_Interval.index
    ABC_Level_Interval_Full = volset_df.merge(ABC_Level_Interval, on='key').set_index('Time_Interval')
    logger.info(f"interval data size: {len(ABC_Level_Interval_Full)}")
    input_features_dict = {'L1_BidSize':'sum','Tot_BidSize':'sum','L1_AskSize':'sum','Tot_AskSize':'sum',
                       'L1_BidSize_perc':'mean','L1_Asksize_perc':'mean','Normalised_Order_Book_Imbalance':'mean',
                        'Level1_Spread':'mean','Level1_Spread_Perc':'mean'}

    rolling_windows = ['15T','1h','24h','120h']

    standard_scaler = preprocessing.StandardScaler()

    if create_labels == False:
        Cols_to_keep = ['Vol']
        ABC_Level_Interval_Full = ABC_Level_Interval_Full[Cols_to_keep]
        ABC_Feature_Cleaned = create_rolling_features(ABC_Level_Interval_Full, ABC_full_df, rolling_windows, input_features_dict)

        np_scaled = standard_scaler.fit_transform(ABC_Feature_Cleaned)
        ABC_Feature_Cleaned_Scaled = pd.DataFrame(np_scaled, columns=ABC_Feature_Cleaned.columns, index = ABC_Feature_Cleaned.index)
        logger.info(f"All features created, data size: {len(ABC_Feature_Cleaned_Scaled)}")
        return ABC_Feature_Cleaned_Scaled
    else:
        tqdm.pandas()
        ABC_Level_Interval_Full = ABC_Level_Interval_Full.progress_apply(lambda row: get_cost_dataframe(row, Bidsizes_levels, Bidprices_levels,'Vol'), axis=1)
        Cols_to_keep = ['Vol','Selling_Cost_Label','Slippage']
        ABC_Level_Interval_Full = ABC_Level_Interval_Full[Cols_to_keep]
        ABC_Feature_Label_Cleaned = create_rolling_features(ABC_Level_Interval_Full, ABC_full_df, rolling_windows, input_features_dict)
        
        X = ABC_Feature_Label_Cleaned.drop(columns=['Selling_Cost_Label','Slippage'])
        np_scaled = standard_scaler.fit_transform(X)
        X_scaled = pd.DataFrame(np_scaled, columns=X.columns, index = X.index)

        X_scaled[['Selling_Cost_Label','Slippage']] = ABC_Feature_Label_Cleaned[['Selling_Cost_Label','Slippage']]
        logger.info(f"All features created, data size: {len(X_scaled)}")
        
        return X_scaled
        

def run_prediction(features_df, path_to_trained_model, path_to_prediction_output, path_to_evaluation_output, config, label=False):

    model_scoring = config['prediction']['model_scoring']
    model_file = read_jsonpickle(path_to_trained_model)

    if type(model_file)==dict:
        model = model_file[model_scoring] 
    else:
        model = model_file


    if label==True:
        X = features_df.drop(columns=['Selling_Cost_Label','Slippage'])
        y_true = features_df['Selling_Cost_Label']
        y_pred = model.predict(X)
        pred_eval =  {'mse': metrics.mean_squared_error(y_true,y_pred),
                    'mae': metrics.mean_absolute_error(y_true, y_pred),
                    'explained_var': metrics.explained_variance_score(y_true, y_pred),
                    'r2': metrics.r2_score(y_true, y_pred)}
        features_df['y_pred'] = y_pred
        write_json(path_to_evaluation_output, pred_eval)

    else:
        y_pred = model.predict(features_df)
        features_df['y_pred'] = y_pred

    features_df.to_csv(path_to_prediction_output)

def train_saveable_model(features_df, path_to_saveable_model, path_to_evaluation_output, config):

    train_config = config['training']

    X = features_df.drop(columns=['Selling_Cost_Label','Slippage'])
    y = features_df['Selling_Cost_Label']

    X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size = 0.25, random_state = 42)
    
    RegModels = RegressionModels(X_train, y_train, train_config['scoring'], train_config['reg_param_scoring'], train_config['cv'])

    models_dct = {}
    for model in train_config['models']: 
        logger.info(f"start to find best param for : {model}")   
        mod_dct = RegModels.find_best_reg_param(model)
        models_dct.update(mod_dct)
    
    best_models = RegModels.get_best_regressor(models_dct)

    model = best_models['mean_mse']
    y_pred = model.predict(X_test)
    pred_eval =  {'mse': metrics.mean_squared_error(y_test,y_pred),
                'mae': metrics.mean_absolute_error(y_test, y_pred),
                'explained_var': metrics.explained_variance_score(y_test, y_pred),
                'r2': metrics.r2_score(y_test, y_pred)}
    # features_df['y_pred'] = y_pred

    write_jsonpickle(path_to_saveable_model, best_models)
    write_json(path_to_evaluation_output, pred_eval)




    

