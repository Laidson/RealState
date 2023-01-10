import streamlit as st
import pandas as pd


class DataManipulation:

    """df have to contan columns:
        PRICE: numpy.int64 | imovel price
        DAYS ON MARKET: numpy.float64
        $/SQUARE FEET: numpy.float64 | squarefeet mesure
        """

    def __init__(self) -> None:
        pass
    
    def upload_file_csv(self):
        uploaded_file = st.file_uploader('Choose a file')
        if uploaded_file is not None:
            #read csv
            df = pd.reade_csv(uploaded_file)
        return df
    
    def convert_df(self, df):
        
        df['LONGITUDE'] = df['LONGITUDE'].astype(str)
        df['LATITUDE'] = df['LATITUDE'].astype(str)
        df['ZIP OR POSTAL CODE'] = df['ZIP OR POSTAL CODE'].astype(str)
        df['SOLD DATE'] = pd.to_datetime(df['SOLD DATE'], format="%m/%d/%Y")

        return df.to_csv(index=False).encode('utf-8')