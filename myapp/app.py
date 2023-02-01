# start with streamlit |terminal: streamlit run myapp/app.py |https://medium.com/geekculture/how-to-run-your-streamlit-apps-in-vscode-3417da669fc
#emoji https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/
import os
import sys
import json
import numpy as np
import streamlit as st 
import pandas as pd
from uuid import uuid4

from src.etl import DataManipulation
from src.charts import Charts
from src.newfeatures import NewFeatures
from src.metrics import MetricCalulation

sys.path.append(os.path.normpath(os.getcwd() + "/src"))
sys.path.append(os.path.normpath(os.getcwd() + "/src/model"))

from src.model.step_04 import ModelPredict
# from src.model.ml_settings import MlSettings





st.title(':house_with_garden: Real State Machine Learning app :robot_face:')
st.markdown('The purpose of this app is to provide summary stats :bar_chart: based on ML data search.')
st.markdown('#### {0} :point_down:'.format('Upload a csv file'))

uploaded_file = st.file_uploader('Choose a file')
# if uploaded_file is not None:
#     #read csv
#     df = pd.read_csv(uploaded_file)
#     df = DataManipulation().convert_df(df)
#     df.to_csv('data/file.csv')
    #uploaded_file.to_csv('data/file.csv')
    

#TODO retirar df abixo apenas para fixar o df de base de desenvolvimento
df = pd.read_csv('data/historical/redfin_2023.csv')
#st.write(df.head()) # write first 5 rows (remove after testing)

# METRICS
calcmetric = MetricCalulation()

st.markdown('## Property Metrics :cityscape:')
col1,col2,col3,col4 = st.columns(4)
col1.metric('Total', calcmetric.num_of_propeties(df), help='Number of properties in search')
col2.metric('Avg Price',calcmetric.avg_price(df), help='Average sale price of properties in search')
col3.metric('Avg DOM', calcmetric.avg_dom(df), help='Average days on market of properties in search')
col4.metric('Avg PPSQFT', calcmetric.avg_ppsqft(df), help='Average price per square foot of properties in search')

# CHARTS
chart = Charts()
with st.expander('Charts :bar_chart:', expanded=False):
    st.markdown('## Charts :bar_chart:')

    #GRAFICS
    chart.create_fig_boxplot(df, 'PRICE',"Price Box Plot")
    chart.create_fig_histogram(df, 'DAYS ON MARKET',"Days on Market Histogram")
    chart.create_fig_histogram(df, '$/SQUARE FEET',"Price per SQFT Histogram") 

# FEATURES
feature = NewFeatures()
df_features = df.copy()

df_features['ratio_sqft_bd'] = feature.ratio_square_feet_dedroom(df_features)
df_features['ratio_lot_sqft'] =  feature.ratio_lot_square_feet(df_features)

df_features['additional_bd_opp'] = df_features.apply(lambda x: feature.additional_bedroom_opportunity(x), axis=1)
df_features['adu_potential'] = df_features.apply(lambda x: feature.adu_potential(x), axis=1)
#st.dataframe(df_features)

# NEW FETURE TABLE EXPORT
# OPORTUNITIES
etl = DataManipulation()
with st.expander('Oportunities :dizzy:', expanded=False):
    st.markdown('## Oportunities :dizzy:')
    col1, col2 = st.columns(2)
    col1.metric('Total Add Bd',len(calcmetric.add_bd(df_features)), help='Number of properties with additonal bedroom opportunity')
    col2.metric('Total ADU', len(calcmetric.add_adu(df_features)), help='Number of properties with ADU potential')

    st.markdown("#### Featurized Dataset")
    st.write(df_features)

    # convert featurized dataset to csv
    csv = etl.convert_df(df_features)

    st.download_button(
        "Download 🔽",
        csv,
        "property_dataset.csv",
        "text/csv",
        key='download-csv'
    )

# ML Predictions
st.markdown('## ML forecasting :robot_face:')
text_inputs = [
    {'label': 'SALE TYPE','tag': 'SALE TYPE','value':'MLS Listing'},
    {'label': 'PROPERTY TYPE','tag': 'PROPERTY TYPE','value':'Townhouse'},
    {'label': 'STATE OR PROVINCE','tag': 'STATE OR PROVINCE','value':'NY'},
    {'label': 'BEDS','tag': 'BEDS','value':'2'},
    {'label': 'BATHS','tag': 'BATHS','value':'2.5'},
    {'label': 'STATUS','tag': 'STATUS','value':'Active'},
    {'label': 'NEXT OPEN HOUSE START TIME','tag': 'NEXT OPEN HOUSE START TIME','value':'No_data'},
    {'label': 'SOURCE','tag': 'SOURCE','value':'REBNY'},
    {'label': 'FAVORITE','tag': 'FAVORITE','value':'N'},
    {'label': 'INTERESTED','tag': 'INTERESTED','value':'Y'},
    {'label': 'SQUARE FEET','tag': 'SQUARE FEET','value':'1733.42'},
    {'label': 'LOT SIZE','tag': 'LOT SIZE','value':'1800.0'},
    {'label': 'YEAR BUILT','tag': 'YEAR BUILT','value':'1930.0'},
    {'label': 'DAYS ON MARKET','tag': 'DAYS ON MARKET','value':'20'},
    {'label': '$-SQUARE FEET','tag': '$-SQUARE FEET','value':'1018.22'},
    {'label': 'HOA-MONTH','tag': 'HOA-MONTH','value':'1641.81'},
    {'label': 'PRICE','tag': 'PRICE','value':'595000'}
    ]

filter_tag = st.selectbox('CHOSSE THE TARGET:', 
                                ['SALE TYPE','PROPERTY TYPE','STATE OR PROVINCE','BEDS','BATHS','STATUS'
                                ,'NEXT OPEN HOUSE START TIME','SOURCE','FAVORITE', 'INTERESTED'
                                ,'SQUARE FEET','LOT SIZE','YEAR BUILT','DAYS ON MARKET','$-SQUARE FEET'
                                ,'HOA-MONTH','PRICE'])

## -----------------------------                                

with st.expander('Insert the House Characteritcs :arrow_down_small:'):
    
    for input in text_inputs:
        if input['label'] != filter_tag:
            input['value'] = st.text_input(input['label'], input['value'])\
                #.validate(lambda value: value if value is not None and value.strip() != "" else st.error("This field is required"))
                    
    
    if st.button('SUBMMIT'):
        filtered_inputs = [input for input in text_inputs if input['label'] != filter_tag]
        dict_iputs = {input['label']:input['value'] for input in text_inputs if input['label'] != filter_tag}
        dict_iputs['ID_REQUEST'] = str(uuid4())[:6]
        st.success('Request submmited! : {}'.format(dict_iputs))

        prediction = ModelPredict(dict_iputs=dict_iputs, target=filter_tag).main()

        st.success(prediction)
     
        
        #save in a json file
        # with open("text_inputs.json", "w") as f:
        #     json.dump(filtered_inputs, f)
        # st.success("Saved to text_inputs.json")



           

    # #forecasting dataset
    # df_prediction = pd.DataFrame()

    # # convert forecasting dataset to csv
    # csv = etl.convert_df(df_prediction)
    # st.download_button(
    #     "Download 🔽",
    #     csv,
    #     "property_dataset.csv",
    #     "text/csv",
    #     key='download-csv'
    # )

df = pd.DataFrame(
    np.random.randn(10, 2) / [20, 20] + [40.71, -74.00],
    columns=['lat', 'lon'])

st.map(df)