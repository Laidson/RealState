import pandas as pd
import os 
import glob
import shutil
from ml_settings import MlSettings


class MLDataInput:
    
    
    def __init__(self) -> None:
        self.df:pd.DataFrame()
        self.source = 'data'
        pass

    def upload_historical_data(self):
        #TODO change for a database connection
        input_dir = f'data/input/{MlSettings.PROJECT_NAME}'
        param_dir = f'working/{MlSettings.PROJECT_NAME}'
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


        return df


    def agg_csv_file(self):
        agg_df = pd.DataFrame()

        df_list = glob.glob(f"{self.source}" + "/*.csv")
        for file in df_list:
            agg_df.append(file)
       
        return agg_df
    
    def remove_leaky_fetures(self):
        #improve this function base on the correlatio value among the features
        col = ['SALE TYPE', 'SOLD DATE', 'PROPERTY TYPE', 'CITY',
       'STATE OR PROVINCE', 'ZIP OR POSTAL CODE', 'PRICE', 'BEDS',
       'BATHS', 'LOCATION', 'SQUARE FEET', 'LOT SIZE', 'YEAR BUILT',
       'DAYS ON MARKET', '$/SQUARE FEET', 'HOA/MONTH', 'STATUS',
       'NEXT OPEN HOUSE START TIME', 'SOURCE', 
       'MLS#', 'FAVORITE', 'INTERESTED', 'LATITUDE', 'LONGITUDE']

        df_col = self.df.columns.values

        deleted_col = list(set(df_col)-set(col))
                
        return self.df[col]

    def cleaning_sinal(self):

        if MlSettings.SEP_DOLLAR:
            #For every column in df, if the column contains a $, make a new column with the value without the $
            for col in self.df.columns:
                if '$' in self.df[col].to_string():
                    self.df[col + '_no_dollar'] = self.df[col].srt.replace('$','').str.replace(',','')
        
        if MlSettings.SEP_PERCENT:
            for col in self.df.column:
                if '%' in self.df[col].to_string():
                    self.df[col + '_no_percent'] = self.df[col].str.replace('%','').str.replace(',','')




    def main(self):        
        self.df = self.upload_historical_data()
        self.df = self.remove_leaky_fetures()
        self.cleaning_sinal()

        pass

