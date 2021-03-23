import os
import numpy as np
import pandas as pd
import collections 
import datetime
import json
import jsonpickle
import time
import logging
from tqdm import tqdm
import click 
from util import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option('--input_dir_path', '-i', 'input_dir_path', help = 'path to input directory. If no path is specified, input folder is set to input_data')
@click.option('--prediction_output', '-p', 'path_to_prediction_output', help = 'Location of csv prediction output, default to output_data/pred_output.csv')
@click.option('--train_model_path', '-t', 'path_to_trained_model', help = 'Location of model file, default to model/best_model_rf.json')
@click.option('--fe_output', '-fe', 'path_to_feature_output', help = 'Location of csv feature output file, default to output_data/feat_output.csv')
@click.option('--eval_output', '-ev', 'path_to_evaluation_output', help = 'Location of evaluation json output file, default to output_data/eval_output.json.')
@click.option('--mode', '-m', 'running_mode', type=click.Choice(['PREDICTION', 'TRAINING','PREDICTION_ONLY','TRAINING_ONLY']), help = 'Select mode of the worker: PREDICTION (wih FE), or TRAINING (with FE), PREDICTION_ONLY, TRAINING_ONLY')
@click.option('--create_labels', '-l', is_flag=True, help = 'Choose creating labels for model evaluation')
@click.option('--config_path', '-c', 'config_path', help = 'Location of config file - default to config.json')
def run_model(running_mode, create_labels, input_dir_path=None, path_to_prediction_output=None, path_to_trained_model=None, path_to_feature_output=None, path_to_evaluation_output=None, config_path=None):
    total_start = time.time()

    input_dir_path = input_dir_path or 'input_data'
    path_to_prediction_output =  path_to_prediction_output or 'output_data/pred_output.csv'
    path_to_trained_model = path_to_trained_model or 'model/best_model_rf.json'
    path_to_feature_output =  path_to_feature_output or 'output_data/feat_output.csv'
    path_to_evaluation_output = path_to_evaluation_output or 'output_data/eval_output.json'
    running_mode = running_mode
    label = True if create_labels else False
    config_path = config_path or 'config.json'

    logger.info(f"create label:{label}")
    logger.info(f"check config:{config_path}")

    config = read_json(config_path)
    input_df = parse_input(input_dir_path, config)
    if running_mode == 'PREDICTION': #runs both FE and PREDICTION in order
        #RUN FE FIRST
        start_t_e = time.time()

        # logger.info(f"Input feature extraction dictionary size: {len(predict_input_dct['parsed'])}")

        # features_dct = run_feature_extraction(predict_input_dct, features_list)
        features_df = run_feature_extraction(input_df, label, config)
        features_df.to_csv(path_to_feature_output)
        end_t_e = time.time()
        logger.info(f"Feature Extraction Execution time: {round((end_t_e-start_t_e),1)}")

        #THEN RUN PREDICTION
        start_t = time.time()
        run_prediction(features_df, path_to_trained_model, path_to_prediction_output, path_to_evaluation_output, config, label)
        end_t = time.time()
        print('Prediction Execution time:', round((end_t - start_t), 1))  

    elif running_mode == "TRAINING": 
        start_t_e = time.time()

        features_df = run_feature_extraction(input_df, label, config)
        features_df.to_csv(path_to_feature_output)
        end_t_e = time.time()
        logger.info(f"Feature Extraction Execution time: {round((end_t_e-start_t_e),1)}")
        
        start_t = time.time()
        train_saveable_model(features_df, path_to_trained_model, path_to_evaluation_output, config)
        end_t = time.time()
        print('Training Execution time:', round((end_t-start_t),1))

    elif running_mode == "TRAINING_ONLY": 

        start_t = time.time()
        features_df = pd.read_csv(path_to_feature_output)
        features_df = features_df.set_index('Unnamed: 0')
        features_df.index = features_df.index.rename('Time_Interval')

        train_saveable_model(features_df, path_to_trained_model, path_to_evaluation_output, config)
        end_t = time.time()
        print('Training Execution time:', round((end_t-start_t),1))

    elif running_mode == 'PREDICTION_ONLY': #runs both FE and PREDICTION in order
        #THEN RUN PREDICTION
        start_t = time.time()
        features_df = pd.read_csv(path_to_feature_output)
        features_df = features_df.set_index('Unnamed: 0')
        features_df.index = features_df.index.rename('Time_Interval')
        run_prediction(features_df, path_to_trained_model, path_to_prediction_output, path_to_evaluation_output, config, label)
        end_t = time.time()
        print('Prediction Execution time:', round((end_t - start_t), 1))  

    else:
        logger.error('Mode selected not working') 
        raise Exception("Mode selected not working. Refer to --help file for details of the accepted running modes")


if __name__ == "__main__":
    run_model()