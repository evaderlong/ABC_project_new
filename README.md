

## Requirements
python version 3.6.5
Please see package requirements in txt file. 
## Intro
This module predicts the selling cost of a security in a future time window. 
Input: it takes 2 input trading datassets (by minute and hour)
Config: 
- General: input data names 
- FE: simulated volumes and predicted time window 
- Training: types of models and evaluation metrics to choose from 
- Prediction: evaluation metric if dataset has labels

Training: (with Feature Extraction)
- by default it produces best model for each eval metric. model in 'model/best_models_vtest.json'
- the best model is RF across eval metrics, a rf model is also created 'model/best_model_rf.json'
- it also produces a evaluation json file
- it produces a FE output file 

Prediction: (with Feature Extraction)
- produces a Prediction output file and a FE output file
- If data has prices data for the predicted timee window, we can choose to create labels and get an evaluation json file.

## Steps
1. Run Prediction + Feature Extraction, see --help for arguments: 
```bash
$ python3 run.py -i <input data folder> -m 'PREDICTION' -p <data path for prediction output> -t <'model/best_models_vtest.json' or 'model/best_model_rf.json'> -l <flag for create label, used for training and predictions evaluation if data has predicted time window prices, otherwise do not specify -l in command> -fe <data path for feature engineering output>
```
2. Run Prediction Only, mode is 'PREDICTION_ONLY', also specify the FE output file for prediction. 
3. Same for Training. 