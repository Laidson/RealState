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
from ml_settings import MlSettings
from step_01 import MLDataInput


class ModelPredict:

    mldatainput = MLDataInput()

    price_input = { 'ID_REQUEST':1, #new column
                    'SALE TYPE':'MLS Listing',
                    'PROPERTY TYPE':'Townhouse',
                    'STATE OR PROVINCE':'NY',
                    'BEDS':'3.0', ##!!
                    'BATHS':'2.0', ##!!
                    'STATUS':'Active',
                    'NEXT OPEN HOUSE START TIME':'No_data',
                    'SOURCE':'BNYMLS',
                    'FAVORITE':'N',
                    'INTERESTED':'Y',
                    'SQUARE FEET':1733.429185,
                    'LOT SIZE':2500,
                    'YEAR BUILT':1925,
                    'DAYS ON MARKET':6,
                    '$-SQUARE FEET':1018.227468,
                    'HOA-MONTH': 1641.819383
                    }
                    #1,5,1,4,3,1,2,1,1,1,1,1,1,2,-0.04853577,-0.37495407,-0.053503655,-0.20085384,-0.42407528,-0.2701332
    def __init__(self) -> None:

        self.target = 'PRICE'
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
        

        # treataments as the trainin and test data bellow--->>>
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
        pd.DataFrame(to.xs).to_csv(f'{self.PARAM_DIR}/requests/ID{index}_{self.target}_request.csv')
        row = to.items.iloc[0]
        to.decode_row(row)
      
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
        peridct = self.model_perdiction(request)

if __name__ == "__main__"        :
    ModelPredict().main()