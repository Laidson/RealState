#system
import glob
import traceback

# data prep stack
import pandas as pd
import numpy as np

#TabNet
from fast_tabnet.core import *
#FastAi
from fastai.tabular.all import *

#LazyPredict
import lazypredict
from lazypredict.Supervised import LazyRegressor
from lazypredict.Supervised import LazyClassifier

#Fit an xgboost model
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score

#internal imports
from ml_settings import MlSettings
from step_01 import MLDataInput



class TrainModel:

    def __init__(self) -> None:
        self.PARAM_DIR = f'working/{MlSettings.PROJECT_NAME}'
        self.targer_list = list()
        
        # info from step 01
        self.dls_dict = MLDataInput.DLS_DICT
        self.df_dls = MLDataInput.DF_DLS
    
    def get_target_names(self):

        name_list = list()
        file_names = glob.glob(f'{self.PARAM_DIR}/y_train_*.csv')
        
        for fn in file_names:
            name = fn.split('_')[-1].split('.')[0]
            name_list.append(name)
        
        self.targer_list = name_list

    def get_xy_train_test_df(self, target):

        X_train = pd.read_csv(f'{self.PARAM_DIR}/X_train_{target}.csv')
        X_test = pd.read_csv(f'{self.PARAM_DIR}/X_test_{target}.csv')
        y_train = pd.read_csv(f'{self.PARAM_DIR}/y_train_{target}.csv')
        y_test = pd.read_csv(f'{self.PARAM_DIR}/y_test_{target}.csv')     
        
        return {'X_train':X_train, 'X_test':X_test, 'y_train':y_train, 'y_test':y_test}

    def model_selection(self, target ,X_train, X_test, y_train, y_test):

        if MlSettings.REFRESSOR == True:
            try:
                reg = LazyRegressor(verbose=2
                                    ,ignore_warnings=False
                                    ,custom_metric=None)
                models, predictions = reg.fit(X_train, X_test, y_train, y_test)
                print(f'Project: {MlSettings.PROJECT_NAME}')
                print(f'Target: {target}')

                target_std = round(y_train.std()[0], 3)
                print(f'Target Standard Deviation: {target_std}')
                print(models)

                models['project_name'] = MlSettings.PROJECT_NAME
                models['target'] = target
                models['target_std'] = target_std

                #rename index of 
                models.to_csv(f'{self.PARAM_DIR}/regression_results_{target}.csv', mode='a', header=True, index=True)#TODO create a new folder
            except:
                print('Issue during lazypredict analysis')
    
    def tabnel_model_selection(self, target):
        to = self.df_dls[target]
        dls = self.dls_dict[target] 

        model_name = 'tabnet'
        learn = None
        i = 0
        while True:
            try:
                del learn
            except:
                pass

            try:
                learn = 0
                #can you please, describe the TabNetModel() inputs?
                model = TabNetModel(get_emb_sz(to), len(to.cont_names), 
                                    dls.c, n_d=64, n_a=64, n_steps=5, 
                                    virtual_batch_size=256)

                # save the best model so far, determined by early stopping
                cbs = [SaveModelCallback(monitor='_rmse', 
                                        comp=np.less, 
                                        fname=f'{model_name}_{MlSettings.PROJECT_NAME}_{target}_best'), 
                                        EarlyStoppingCallback()]
                print(cbs)
                learn = Learner(dls,
                                model,
                                loss_func=MSELossFlat(),
                                metrics=rmse,
                                cbs=cbs)
                print(learn)
                if learn != 0:
                    break
                if i > 50:
                    break
            except:
                i += 1
                print('Error in FastAI TabNet')
                traceback.print_exc()
                continue

        try:
            """This value should be used when training a model with a learning rate. 
               This value can be used in conjunction with other parameters such as 
               the number of epochs and the batch size to optimize the model for 
               the best performance."""
            x = learn.lr_find()
        except:
            pass
        
        
        return {'x':x, 'i':i, 'learn':learn}

    def auto_learning_setting(self, x, i, learn):

        if MlSettings.AUTO_AJUST_LEARNING_RATE == False:#True:        
            MlSettings.FASTAI_LEARNING_RATE = x.valley
            print(f'LEARNING RATE: {MlSettings.FASTAI_LEARNING_RATE}')

        try:
            if i < 50:
                learn.fit_one_cicle(20, MlSettings.FASTAI_LEARNING_RATE)
                plt.figure(figsize=(20,20))
                try:
                    ax = learn.show_results()
                    plt.show()
                except:
                    print('Could not show results')
                    pass
        except:
            print('Could not fit model')
            traceback.print_exc()
            pass

    def fit_xgboost_model(self, target, X_train, X_test, y_train, y_test):
        
        if MlSettings.REFRESSOR == True:
            xgb = XGBRegressor()
        else:
            xgb = XGBClassifier
        
        try:
            xgb.fit(X_train, y_train)
            y_pred = xgb.predict(X_test)
            print('XGBoost Predictions vs Actual==========')
            print(pd.DataFrame({'actual': y_test.iloc[:,0].values, 'predicted': y_pred}).head())
            print('XGBoost RMSE: ', np.sqrt(mean_squared_error(y_test, y_pred)))

            #save feature importance plot to file
            plot_importance(xgb)
            plt.title(f'XGBoost Feature Importance for {MlSettings.PROJECT_NAME} | Target : {target}', wrap=True)
            plt.tight_layout()
            plt.show()
            plt.savefig(f'{self.PARAM_DIR}/xgb_feature_importance_{target}.png') #TODO inswet a new folder

            fi_df = pd.DataFrame([xgb.get_booster().get_score()]).T
            fi_df.columns = ['importance']

            #create a column based off the index called feature
            fi_df['feature'] = fi_df.index
            # create a dataframe of feature importance
            fi_df = fi_df[['feature','importance']]
            fi_df.to_csv(f'{self.PARAM_DIR}/xgb_feature_importance_{target}.csv', index=False)#TODO create a folder
          
        except: 
            traceback.print_exc()
            print('XGBoost failed')
              
        


    def main(self):
        self.get_target_names()
        for target in ['DAYS ON MARKET', 'PRICE']:#self.targer_list:
            dfs = self.get_xy_train_test_df(target)
            X_train = dfs['X_train']
            X_test = dfs['X_test'] 
            y_train = dfs['y_train'] 
            y_test = dfs['y_test']

            #LazyRegressor models
            self.model_selection(target, X_train, X_test, y_train, y_test)
            
            # #TabNet models
            # tabnet_vars = self.tabnel_model_selection(target)          
            # #Tabnet - Auto learning set
            # self.auto_learning_setting(x=tabnet_vars['x'],i=tabnet_vars['i'],learn=tabnet_vars['learn'])

            #Fit XGBoost
            self.fit_xgboost_model(target, X_train, X_test, y_train, y_test)
        return
        



