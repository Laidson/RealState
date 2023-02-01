import os
import sys
sys.path.append(os.path.normpath(os.getcwd() + "/src"))

# data manipulation
import pandas as pd
#Fit an xgboost model
from xgboost import XGBRegressor
from xgboost import XGBClassifier
#fatsAI
from fastai.tabular.all import *
from fastai.tabular.core import *
from fastai.tabular.data import *
#internal imports
from src.model.ml_settings import MlSettings
from src.model.step_01 import MLDataInput
# from step_01 import MLDataInput
# from ml_settings import MlSettings


class ModelPredict:

    mldatainput = MLDataInput()

    #'PRICE': 595000,  16356160
    price_input = {'SALE TYPE': 'MLS Listing',
    'PROPERTY TYPE': 'Townhouse',
    'STATE OR PROVINCE': 'NY',
    'BEDS': '2',
    'BATHS': '2.5',
    'STATUS': 'Active',
    'NEXT OPEN HOUSE START TIME': 'No_data',
    'SOURCE': 'REBNY',
    'FAVORITE': 'N',
    'INTERESTED': 'Y',
    'SQUARE FEET': '1733.42',
    'LOT SIZE': '1800.0',
    'YEAR BUILT': '1930.0',
    'DAYS ON MARKET': '20',
    '$-SQUARE FEET': '1018.22',
    'HOA-MONTH': '1641.81',
    'ID_REQUEST': '0a8546'}

    # price_input = {'SALE TYPE':'1',
    #                 'PROPERTY TYPE': '1', 
    #                 'STATE OR PROVINCE': '1', 
    #                 'BEDS': '1',
    #                 'BATHS':'1' ,
    #                 'STATUS': '11', 
    #                 'NEXT OPEN HOUSE START TIME': '1',
    #                 'SOURCE': '1',
    #                 'FAVORITE': '111',
    #                 'INTERESTED': '1', 
    #                 'SQUARE FEET': '1',
    #                 'LOT SIZE': '1',
    #                 'YEAR BUILT': '1',                    
    #                 'DAYS ON MARKET': '1',
    #                 '$-SQUARE FEET': '1',
    #                 'HOA-MONTH': '1',
    #                 'ID_REQUEST': '0a8546'}

                    
    def __init__(self, dict_iputs, target) -> None:

        self.target = 'PRICE' #TODO insert target 
        self.input = self.price_input#TODO dict_iputs
        self.PARAM_DIR = f'working/{MlSettings.PROJECT_NAME}'#TODO insert path for model folders
        self.model = None
        
    def get_categorical_columns(self):
        
        with open(f'{self.PARAM_DIR}/{self.target}_cats.txt', 'r') as file:
            lines = file.readlines()
            # Remove a newline characters from each line
            lines = [line.strip() for line in lines]

        return lines

    def get_continuous_columns(self):

        with open(f'{self.PARAM_DIR}/{self.target}_conts.txt', 'r') as file:
            lines = file.readlines()
            # Remove a newline characters from each line
            lines = [line.strip() for line in lines]

        return lines

    def load_feature_model(self):
        #TODO for now making dor XGboost
        model = XGBRegressor()
        model.load_model(f'{self.PARAM_DIR}/prod/model_select/model_{self.target}.json')
        return model
    
    def creat_model_inputs(self):
        index = [self.price_input['ID_REQUEST']]
        request = pd.DataFrame(self.price_input, index=index)
        del(request['ID_REQUEST'])

        #Open and apply the mask over the request msg
        data = pd.read_json(f'{self.PARAM_DIR}/prod/mask/{self.target}_cat_mask.json').to_dict()
        mask = data[self.target]

        for col, mask_values in mask.items():
            request.loc[:, col] = request.loc[:, col].replace(mask_values)
        
        #convert the request datatype

        dt_types = {
        'PROPERTY TYPE':                   int,
        'SALE TYPE':                       int,
        'STATE OR PROVINCE':               int,
        'BEDS':                            int,
        'BATHS':                           int,
        'STATUS':                          int,
        'NEXT OPEN HOUSE START TIME':      int,
        'SOURCE':                          int,
        'FAVORITE':                        int,
        'INTERESTED':                      int,
        'SQUARE FEET':                   float,
        'LOT SIZE':                      float,
        'YEAR BUILT':                    float,
        'DAYS ON MARKET':                float,
        '$-SQUARE FEET':                 float,
        'HOA-MONTH':                     float,
        'PRICE':                         float
        }
        del(dt_types[self.target])
        request = request.astype(dtype=dt_types)

        # treataments as the trainin and test data bellow--->>>
        try:
            r = {'df': df_shrink(request)
                            ,'procs':[Categorify, FillMissing, Normalize]
                            ,'cats': self.get_categorical_columns()
                            ,'conts': self.get_continuous_columns()
                            ,'target':self.target}
                            #,'splits':splits}'''

            result = TabularPandas(df=r['df']
                                ,procs=r['procs']
                                ,cat_names=r['cats']
                                ,cont_names=r['conts']
                                ,y_names=r['target']
                                ,y_block=RegressionBlock()
                                    )
            
            #TODO inspect the data after create the tabularpandas
            
            to = result.copy()
            #pd.DataFrame(to.xs).to_csv(f'{self.PARAM_DIR}/requests/ID{index}_{self.target}_request.csv')
            row = to.items.iloc[0]
            to.decode_row(row)
        except: 
            pass
        request.to_csv(f'{self.PARAM_DIR}/requests/ID{index}_{self.target}_request.csv', index=False)
        return request #to.xs
        
    def model_perdiction(self, request):
        prediction = self.model.predict(request)
        print(prediction)
        return prediction


    def main(self):
        #recive feature selection to predict

        #load model by target input
        self.model = self.load_feature_model()

        #make the prediction
        #TODO aplly a mask over categorical fetures
        request = self.creat_model_inputs()      

        # give the prediction result back
        predict = self.model_perdiction(request)

        return f'{self.target}: {predict[0]}'

if __name__ == "__main__"        :
    ModelPredict(dict(),'AA').main()