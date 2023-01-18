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
                    'SALE TYPE':1,
                    'PROPERTY TYPE':1,
                    'STATE OR PROVINCE':1,
                    'BEDS':1,
                    'BATHS':1,
                    'STATUS':1,
                    'NEXT OPEN HOUSE START TIME':'No_data',
                    'SOURCE':1,
                    'FAVORITE':1,
                    'INTERESTED':1,
                    'SQUARE FEET':10,
                    'LOT SIZE':100,
                    'YEAR BUILT':10,
                    'DAYS ON MARKET':1,
                    '$-SQUARE FEET':100,
                    'HOA-MONTH': 1
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
        model.load_model(f'{self.PARAM_DIR}/model_{self.target}.json')
        return model
    
    def creat_model_inputs(self):
        index = [self.price_input['ID_REQUEST']]
        request = pd.DataFrame(self.price_input, index=index)
        del(request['ID_REQUEST'])

        r = {'df': df_shrink(request)
                        ,'procs':[Categorify, FillMissing, Normalize]
                        ,'cats': self.get_categorical_columns()
                        ,'conts': self.get_continuous_columns()
                        ,'target':self.target}
                        #,'splits':splits}

        result = TabularPandas(df=r['df']
                              ,procs=r['procs']
                              ,cat_names=r['cats']
                              ,cont_names=r['conts']
                              ,y_names=r['target']
                              ,y_block=RegressionBlock()
                                )
        
        #TODO inspect the data after create the tabularpandas
        #TODO testar input com dados iguais a um de treino para ver se sofrem os memsmos pre tratamenots.
        to = result.copy()
        pd.DataFrame(to.train.xs).to_csv(f'{self.PARAM_DIR}/request.csv')
        row = to.items.iloc[0]
        to.decode_row(row)

        request_df = pd.read_csv(f'{self.PARAM_DIR}/request.csv')
        request_df.drop(request_df.columns[0], axis=1, inplace=True)
       
        return request_df
        
    def model_perdiction(self, request):
        prediction = self.model.predict(request)
        
        return prediction


    def main(self):
        #recife reature selection to predict
        #load model by target input
        self.model = self.load_feature_model()        
        #make the prediction
        request = self.creat_model_inputs()

        #TODO delete #self.mldatainput.create_tabular_dataset_object(result_dict=request, target=self.target)

        peridct = self.model_perdiction(request)

        # give the prediction result back

if __name__ == "__main__"        :
    ModelPredict().main()