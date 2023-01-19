import pandas as pd
import os 
import glob
import shutil
import re
from ml_settings import MlSettings

from fastai.tabular.all import *
from fastai.tabular.core import *
from fastai.tabular.data import *


class MLDataInput:

    DLS_DICT = dict()
    DF_DLS = dict()
    
    
    def __init__(self) -> None:
        self.imput_dir = f'data/input/{MlSettings.PROJECT_NAME}'
        self.param_dir = f'working/{MlSettings.PROJECT_NAME}'
        self.TARGET = list()

        self.df : pd.DataFrame()
        self.categorical : list()
        self.continuous : list()
        self.source = 'data'
        pass

    def upload_historical_data(self):
        #TODO change for a database connection
        input_dir = self.imput_dir
        param_dir = self.param_dir
        TAGET = ''
        PARAM_DIR = param_dir
        print(f'param_dir: {param_dir}')
        
        if not os.path.exists(param_dir):
            os.makedirs(param_dir)
        #rename any file in param_dir/file that ends with csv to data.csv
        for file in os.listdir(input_dir):
            if file.endswith('.csv'):
                print('csv!!!')
                if 'classification_results' not in file and 'regression_results' not in file:
                    shutil.copy(f'{input_dir}/{file}', f'{param_dir}/data.csv')                         
        try:
            df = pd.read_csv(f'{param_dir}/data.csv', nrows=MlSettings.SAMPLE_COUNT)
        except:
            print(f'Missing file named data.csv in {param_dir}')
        
        try:
            df = df.rename(columns=lambda x:re.sub('/','-', x))
        except: pass

        return df


    def agg_csv_file(self):
        agg_df = pd.DataFrame()

        df_list = glob.glob(f"{self.source}" + "/*.csv")
        for file in df_list:
            agg_df.append(file)
       
        return agg_df
    
    def remove_leaky_fetures(self):
        #improve this function base on the correlatio value among the features
        col = ['SALE TYPE', 'PROPERTY TYPE', 'CITY',
       'STATE OR PROVINCE', 'PRICE', 'BEDS',
       'BATHS', 'LOCATION', 'SQUARE FEET', 'LOT SIZE', 'YEAR BUILT',
       'DAYS ON MARKET', '$-SQUARE FEET', 'HOA-MONTH', 'STATUS',
       'NEXT OPEN HOUSE START TIME', 'SOURCE', 
       'MLS#', 'FAVORITE', 'INTERESTED']
        
        df_col = self.df.columns.values

        deleted_col = list(set(df_col)-set(col))
                
        return self.df[col]

    def cleaning_sinal(self):

        if MlSettings.SEP_DOLLAR:
            #For every column in df, if the column contains a $, make a new column with the value without the $
            for col in self.df.columns:
                print(col)
                if '$' in self.df[col].to_string():
                    self.df[col + '_no_dollar'] = self.df[col].srt.replace('$','').str.replace(',','')
        
        if MlSettings.SEP_PERCENT:
            for col in self.df.columns:
                if '%' in self.df[col].to_string():
                    self.df[col + '_no_percent'] = self.df[col].str.replace('%','').str.replace(',','')

    def select_target_columns(self):
        target = ''
        target_str = ''
        #The column closest to the end isPARAM_DIR the target variable that can be represented as a float is the target variable
        targets = []
        #Loop through every possible target column (Continuous)
        for i in range(len(self.df.columns)-1,0,-1):
            try:
                self.df[self.df.columns[i]] = self.df[self.df.columns[i]].astype(float)
                target = self.df.columns[i]
                target_str = target.replace('/','-')
            except:
                continue
            
            #Save target feture on self.TARGET list
            self.TARGET.append(target_str)

    def create_ml_config_files(self, target):
        PARAM_DIR = f'working/{MlSettings.PROJECT_NAME}'
        #Create project config files if they don't exist.
        if not os.path.exists(self.param_dir):
            #creat param_dir
            os.mkdir(PARAM_DIR)

        # CATEGORICAL FEATURE FILE    
        if not os.path.exists(f'{PARAM_DIR}/{target}_cats.txt'):
            #creat param_dir
            with open(f'{PARAM_DIR}/{target}_cats.txt', 'w') as f:
                f.write('')
        
        # CONTINUOUS FEATURE FILE 
        if not os.path.exists(f'{PARAM_DIR}/{target}_conts.txt'):
            #create param_dir
            with open(f'{PARAM_DIR}/{target}_conts.txt', 'w') as f:
                f.write('')
        
        # DELETED FEATURE COLUMNS
        if not os.path.exists(f'{PARAM_DIR}/{target}_cols_to_delete.txt'):
            #create param_dir
            with open(f'{PARAM_DIR}/{target}_cols_to_delete.txt', 'w') as f:
                f.write('')

        # Setp 02
        #  Storage the def model_selection() results
        if not os.path.exists(f'{PARAM_DIR}/select_models/reg_results/'):
            os.makedirs(f'{PARAM_DIR}/select_models/reg_results/')    
        
        #Storage XGBoost results
        if not os.path.exists(f'{PARAM_DIR}/xgboost/'):
            os.makedirs(f'{PARAM_DIR}/xgboost/')
            
        # Step_04
        # storage the requests predictions messages
        if not os.path.exists(f'{PARAM_DIR}/requests/'):
            os.makedirs(f'{PARAM_DIR}/requests/')



   

    def auto_detect_cat_conts_variavles(self, target):
        #Auto detect categorical and continuous variables
        likely_cat = {}
        for var in self.df.columns:
            likely_cat[var] = 1.*self.df[var].nunique()/self.df[var].count() < 0.05 #or some other threshold
        
        self.categorical = [var for var in self.df.columns if likely_cat[var]]
        self.continuous = [var for var in self.df.columns if not likely_cat[var]]
        cats = [var for var in self.df.columns if likely_cat[var]]
        conts = [var for var in self.df.columns if not likely_cat[var]]
        
        # Remove targets from features lists
        try:
            self.categorical.remove(target)            

        except: pass

        try:
            self.continuous.remove(target)
        except: pass

        #Populate categorical and continuous lists
        with open(f'{self.param_dir}/{target}_cats.txt', 'w') as f:
            for i in self.categorical:
                f.write("%s\n" % i)

        #saving a Continuous list on txt file
        with open(f'{self.param_dir}/{target}_conts.txt', 'w') as f:
            for i in self.continuous:
                f.write("%s\n" % i)
                

        print(f'{target}:{type(self.df[target][0])}')

    
    def read_cat_conts_feature_lists(self, target):
        """
            target: target feature name that you want to read the files
        """
        if MlSettings.VARIABLE_FILES == True:

            with open(f'{self.param_dir}/{target}_cats.txt', 'r') as f:
                cat_list = f.read().splitlines()

            with open(f'{self.param_dir}/{target}_conts.txt', 'r') as f:
                conts_list = f.read().splitlines()
        
        return cat_list, conts_list
    
    def update_txt_cat_conts_features(self, target):

        #Populate categorical and continuous lists
        with open(f'{self.param_dir}/{target}_cats.txt', 'w') as f:
            for i in self.categorical:
                f.write("%s\n" % i)

        #saving a Continuous list on txt file
        with open(f'{self.param_dir}/{target}_conts.txt', 'w') as f:
            for i in self.continuous:
                f.write("%s\n" % i)



    def find_break_point_feature(self, target):
        """
            The main goal of finding continuous variables to find breakpoints in a forecasting data frame 
            is to identify changes in trends or patterns in the data that may have an impact on the accuracy of predictions. 
            Continuous variables can be used to identify potential changes in the data, such as the emergence of a new trend 
            or the end of an existing one. By finding these breakpoints, forecasts can be more accurately adjusted to take into account these changes, 
            thus improving the accuracy of predictions.
        """
        
        df = self.df.copy()
        procs = [Categorify, FillMissing, Normalize]
        df = df[0:MlSettings.SAMPLE_COUNT]
        splits = RandomSplitter()(range_of(df))
        print((len(self.categorical)) + len(self.continuous), len(df.columns))

        if MlSettings.ENABLE_BREAKPOINT == True:
            temp_procs = [Categorify, FillMissing]
            print('Looping through continuous variables to find breakpoint')
            cont_list = list()
            for cont in self.continuous:
                focus_cont = cont
                cont_list.append(cont)

                try:
                    to = TabularPandas(df=df
                                    ,procs=procs
                                    ,cat_names=self.categorical
                                    ,cont_names=cont_list
                                    ,y_names=target
                                    ,y_block=RegressionBlock()
                                    ,splits=splits)
                    del(to)
                
                except:
                    print('ERROR with: ', focus_cont)
                    #remove focus_cont from list
                    cont_list.remove(focus_cont)
                    continue
            #convert all continuous variable to folats
            for var in cont_list:
                try:
                    df[var] = df[var].astype(float)
                except:
                    print(f'Could not convert {var} to float')
                    cont_list.remove(var)
                    if MlSettings.CONVERT_TO_CAT == True:
                        self.categorical.append(var)
                pass
            self.continuous = cont_list
            print(f'Continuous variables that made the cut : {cont_list}')
            print(f'Categorical variables that made the cut : {self.categorical}')
            
            result = {'df': df_shrink(df)
                        ,'procs':procs
                        ,'cats': self.categorical
                        ,'conts': cont_list #TODO atualizar self.continuous = cont_list?! #self.continuous #TODO talvez cont_list?
                        ,'target':target
                        ,'splits':splits}
            
            self.update_txt_cat_conts_features(target=target)
        
        return result


    def create_tabular_dataset_object(self, result_dict, target):

        to = None

        if MlSettings.REFRESSOR == True:
            try:
                to = TabularPandas(df=result_dict['df']
                                    ,procs=result_dict['procs']
                                    ,cat_names=result_dict['cats']
                                    ,cont_names=result_dict['conts']
                                    ,y_names=result_dict['target']
                                    ,y_block=RegressionBlock()
                                    ,splits=result_dict['splits']
                                        )
            except:
                conts = []
                to = TabularPandas(df=result_dict['df']
                                    ,procs=result_dict['procs']
                                    ,cat_names=result_dict['cats']
                                    ,cont_names=conts
                                    ,y_names=result_dict['target']
                                    ,y_block=RegressionBlock()
                                    ,splits=result_dict['splits']
                                        )
        else:
            try:
                to = TabularPandas(df=result_dict['df']
                                    ,procs=result_dict['procs']
                                    ,cat_names=result_dict['cats']
                                    ,y_names=result_dict['target']
                                    ,splits=result_dict['splits']
                                    )
            except:
                conts = []
                to = TabularPandas(df=result_dict['df']
                                    ,procs=result_dict['procs']
                                    ,cat_names=result_dict['cats']
                                    ,cont_names=conts
                                    ,y_names=result_dict['target']
                                    ,splits=result_dict['splits']
                                        )
        dls = to.dataloaders()
        print(f'Tabular Object size: {len(to)}')
        self.create_mask_dict_for_cat_features(result_dict['df'], to, result_dict['cats'], result_dict['target'])
        try:
            dls.one_batch()

        except:
            print(f'problem with getting one batch of {MlSettings.PROJECT_NAME}')

        #TODO Pensar se é nescessário retirar quando for trabalhar sobre performance do modelo
        #tentativa de acessar to.columns para excluir '_na" columns
        #toma muito tempo, melhor no fututo tratar os dados para evitar o surgimento dessas colunas
        #causado pela presnca de Nan values on the columns
        # to = TabularDataLoaders.from_df(to, cat_names=result_dict['cats'], cont_names=result_dict['conts'], procs=result_dict['procs'])
        # to = self.drop_na_new_columns(to)           

        return {f'df_{target}': to, f'dls_{target}':dls}
    
    def create_mask_dict_for_cat_features(self, df_in, to, cat_list, target):
        #TODO inset target
        #TODO open the cat features from txt file
        #Ex:
        code_dict = dict()
        code_dict[target] = dict()
        df = pd.merge(df_in, to.xs, left_index=True, right_index=True)
        for cat in cat_list:
            dict_mask = df.groupby(f'{cat}_x')[f'{cat}_y'].first().to_dict()        
            code_dict[target].update({cat:dict_mask})

        df_dict = pd.DataFrame.from_dict(code_dict)
        df_dict.to_json()#TODO vreate a folder to save this    
        code_dict.to_json(f'{target}_cat_mask.json')
        return
    
    def drop_na_new_columns(self, to):
        #TODO Pensar se é nescessário retirar quando for trabalhar sobre performance do modelo
        #TODO Tratar data frame antes de fazer treinamentos no modelo
        """
        The TabularPandas function from the fastai.tabular library is used to create a TabularDataLoaders object from a Pandas DataFrame. 
        When creating a new TabularDataLoaders object, the TabularPandas function checks for missing values in the input DataFrame and creates new 
        columns with the suffix "_na" for each column that contains missing values. These new columns are used to keep track of which values in the original columns were missing.
        For example, if you have a column "age" with missing values, the TabularPandas function will create a new column "age_na" that has a value of 1 for rows where the "age" 
        column is missing, and 0 for rows where the "age" column is not missing.
        This is done so that the TabularDataLoaders object can handle missing values properly during the training and prediction process. If you don't want these columns, 
        you can drop them after creating the TabularDataLoaders object or drop them before passing the dataframe to TabularPandas function.        
        """
        df = to.loc[:,~to.columns.str.endswith('_na')]
        return df


    def create_train_test_tabular_object(self, target, info_dict):
        #Built-in split  train/test 80/20 % 

        to = info_dict[f'df_{target}']
        
        X_train, y_train = to.train.xs, to.train.ys.values.ravel()
        X_test, y_test = to.valid.xs, to.valid.ys.values.ravel()

        #Make sure the target isn't in independent columns
        if target in X_train and target in X_test:
            del(X_train[target])
            del(X_test[target])

        #create dataframe from X_train and y_train
        #export tabular object to csv
        pd.DataFrame(X_train).to_csv(f'{self.param_dir}/X_train_{target}.csv', index=False)
        pd.DataFrame(X_test).to_csv(f'{self.param_dir}/X_test_{target}.csv', index=False)
        pd.DataFrame(y_train).to_csv(f'{self.param_dir}/y_train_{target}.csv', index=False)
        pd.DataFrame(y_test).to_csv(f'{self.param_dir}/y_test_{target}.csv', index=False)


    def main(self):        
        self.df = self.upload_historical_data()
        self.df = self.remove_leaky_fetures()
        self.cleaning_sinal()            
        self.df = df_shrink(self.df)

        self.select_target_columns() # create self.TARGET = list()

        # Creat a data frame with each possible TARGET column
        for target in ['DAYS ON MARKET', 'PRICE']: #self.TARGET:#TODO return with all feature list
          
            self.create_ml_config_files(target) #files to save fetire names 
            self.auto_detect_cat_conts_variavles(target)# retirar some features from the self.categorical and self.continuous
            #TODO decide to insert on this line SHUFFLE_DATA or not ?!?

            result_dict = self.find_break_point_feature(target)
            pre_train_test_spit = self.create_tabular_dataset_object(result_dict, target)
            self.DLS_DICT[target] = pre_train_test_spit[f'dls_{target}'] # save the dls target obj on dict 
            self.DF_DLS[target] = pre_train_test_spit[f'df_{target}']
            self.create_train_test_tabular_object(target, pre_train_test_spit) 

        return

    


