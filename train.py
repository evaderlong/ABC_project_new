import logging 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn. linear_model import LinearRegression
from sklearn.svm import SVR, LinearSVR

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np

class RegressionModels(object):
    def __init__(self, X=None, y=None, scoring=None, reg_param_scoring=None, cv=None):
        self.X = X
        self.y = y
        # self.regressors_dct = regressors_dct
        self.scoring = scoring 
        self.cv = cv 
        self.reg_param_scoring = reg_param_scoring

    def find_best_reg_param(self, regressor_name):
        
        # AdaBoost

        try:
            if regressor_name == "LinearSVR":
                param_grid=[{
                    'loss':['epsilon_insensitive', 'squared_epsilon_insensitive'], 'C':[1,0.1, 0.01], 'random_state':[42]}
                ]
                svr=LinearSVR()
                gs = GridSearchCV(svr, param_grid, scoring=self.reg_param_scoring, cv=self.cv)
                gs.fit(self.X, self.y)
                regressor=gs.best_estimator_
                logger.info(regressor_name, regressor)
                
            if regressor_name == "LR":
                lr=LinearRegression()
    #             gs = GridSearchCV(adaboost, param_grid, scoring=scoring, cv=cv)
                lr.fit(self.X, self.y)
                regressor=lr
                logger.info(regressor_name, regressor)
                
            if regressor_name == "AB":
                param_grid=[{
                    'n_estimators':[100, 50, 25], 'learning_rate':[1,0.1, 0.01], 'loss':['linear','square'], 'random_state':[42]}
                ]
                adaboost=AdaBoostRegressor()
                gs = GridSearchCV(adaboost, param_grid, scoring=self.reg_param_scoring, cv=self.cv)
                gs.fit(self.X, self.y)
                regressor=gs.best_estimator_
                logger.info(regressor_name, regressor)

            #RF
            if regressor_name == "RF":
                param_grid=[
                    {'n_estimators':[100, 50, 25], 'max_depth':[50, 100, 150], 'max_features':[2, 3, 4], 'random_state':[42]}
                ]
                rforest = RandomForestRegressor()
                gs = GridSearchCV(rforest, param_grid, scoring=self.reg_param_scoring, cv=self.cv)
                gs.fit(self.X, self.y)
                regressor=gs.best_estimator_
                logger.info(regressor_name, regressor)


            # GB
            if regressor_name == "GB":
                param_grid=[{
                    'n_estimators':[100, 50, 25], 'max_depth':[50, 100, 150], 'random_state':[42]}
                ]
                gboost=GradientBoostingRegressor()
                gs = GridSearchCV(gboost, param_grid, scoring=reg_param_scoring, cv=cv)
                gs.fit(X, y)
                regressor=gs.best_estimator_
                logger.info(regressor_name, regressor)
                
            regressors_dct = {regressor_name: regressor}
            return regressors_dct
        
        except:
            logger.error("Error: No model returned, you need to choose among Adaboost, RandmoForest, GradientBoosting, Linear Regression or Linear SVR")
            regressor = None
            pass


    def get_best_regressor(self, regressors_dct):
            
        best_regressors = {}
        for score_name in self.scoring.keys():
            if 'mean' in score_name:
                best_regressor = None
                regressor_result = {}
                for reg_name, reg_instance in regressors_dct.items():
                    reg_scores = cross_val_score(reg_instance, self.X, self.y, scoring=self.scoring[score_name], cv=self.cv)    
                    logger.info(reg_name, reg_scores)
                    logger.info("Average {} of CV:{}".format(score_name, np.mean(reg_scores)))
                    mean_score = np.mean(reg_scores)
                    regressor_result.update({reg_name:{'score':mean_score,'regressor':reg_instance}})
                logger.info(regressor_result)
                best_regressor = min(regressor_result.items(), key = lambda regname:abs(regname[1]['score']))
                logger.info('\n\n *** BEST MODEL FOUND for {}:{}'.format(score_name, best_regressor[1]['regressor']), '***')    
                best_regressors.update({score_name:best_regressor[1]['regressor']})
        
            else:
                best_score = 0
                best_regressor = None
                for reg_name, reg_instance in regressors_dct.items():
                    reg_scores = cross_val_score(reg_instance, self.X, self.y, scoring=self.scoring[score_name], cv=self.cv)    
                    logger.info(reg_name, reg_scores)
                    logger.info("Average {} of CV:{}".format(score_name, np.mean(reg_scores)))
                    mean_score = np.mean(reg_scores)
                    if mean_score > best_score:
                        best_score = mean_score
                        best_regressor = reg_instance
                logger.info('\n\n *** BEST MODEL FOUND for {}:{}'.format(score_name, best_regressor), '***')
                best_regressors.update({score_name:best_regressor})

        return best_regressors