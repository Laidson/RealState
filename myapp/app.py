# start with streamlit |terminal: streamlit run myapp/app.py |https://medium.com/geekculture/how-to-run-your-streamlit-apps-in-vscode-3417da669fc
#emoji https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/
import os
import sys
import json
import streamlit as st 
import pandas as pd
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
# with st.expander('ML forecasting :robot_face:'):
#     st.markdown('## ML forecasting :robot_face:')

#     #input info for request message
#     st.markdown('Property characteritcs')
#     sales_type = st.text_input('SALE TYPE','MLS Listing')
#     property_type = st.text_input('PROPERTY TYPE','Townhouse')
#     satet = st.text_input('STATE OR PROVINCE','NY')
#     beds = st.text_input('BEDS','3.0')
#     baths = st.text_input('BATHS','2.0')
#     status = st.text_input('STATUS','Active')
#     openhouse_date = st.text_input('NEXT OPEN HOUSE START TIME','No_data')
#     source = st.text_input('SOURCE','BNYMLS')
#     favorite = st.text_input('FAVORITE','N')
#     interested = st.text_input('INTERESTED','Y')
#     sq_feet = st.text_input('SQUARE FEET',1733.429185)
#     lot_size =st.text_input('LOT SIZE',2500)
#     year_build = st.text_input('YEAR BUILT',1925)
#     days_market = st.text_input('DAYS ON MARKET',6)
#     price_sq_feet = st.text_input('$-SQUARE FEET',1018.227468)
#     hoa = st.text_input('HOA-MONTH',1641.81938)
#     price = st.text_input('PRICE',1000.00)

st.markdown('## ML forecasting :robot_face:')
text_inputs = [
    {'label': 'SALE TYPE','tag': 'SALE TYPE','value':''},
    {'label': 'PROPERTY TYPE','tag': 'PROPERTY TYPE','value':''},
    {'label': 'STATE OR PROVINCE','tag': 'STATE OR PROVINCE','value':''},
    {'label': 'BEDS','tag': 'BEDS','value':''},
    {'label': 'STATUS','tag': 'STATUS','value':''},
    {'label': 'NEXT OPEN HOUSE START TIME','tag': 'NEXT OPEN HOUSE START TIME','value':''},
    {'label': 'SOURCE','tag': 'SOURCE','value':''},
    {'label': 'FAVORITE','tag': 'FAVORITE','value':''},
    {'label': 'INTERESTED','tag': 'INTERESTED','value':''},
    {'label': 'SQUARE FEET','tag': 'SQUARE FEET','value':''},
    {'label': 'LOT SIZE','tag': 'LOT SIZE','value':''},
    {'label': 'YEAR BUILT','tag': 'YEAR BUILT','value':''},
    {'label': 'DAYS ON MARKET','tag': 'DAYS ON MARKET','value':''},
    {'label': '$-SQUARE FEET','tag': '$-SQUARE FEET','value':''},
    {'label': 'HOA-MONTH','tag': 'HOA-MONTH','value':''},
    {'label': 'PRICE','tag': 'PRICE','value':''}
    ]

filter_tag = st.selectbox('CHOSSE THE TARGET:', ['SALE TYPE','PROPERTY TYPE','STATE OR PROVINCE','BEDS','STATUS'
                                ,'NEXT OPEN HOUSE START TIME','SOURCE','FAVORITE', 'INTERESTED'
                                ,'SQUARE FEET','LOT SIZE','YEAR BUILT','DAYS ON MARKET','$-SQUARE FEET'
                                ,'HOA-MONTH','PRICE'])

## -----------------------------                                

with st.expander('Insert the House Characteritcs :arrow_down_small:'):
    
    for input in text_inputs:
        if input['label'] != filter_tag:
            input['value'] = st.text_input(input['label'])\
                #.validate(lambda value: value if value is not None and value.strip() != "" else st.error("This field is required"))
                    
    
    if st.button('SUBMMIT'):
        filtered_inputs = [input for input in text_inputs if input['label'] != filter_tag]
        dict_iputs = {input['label']:input['value'] for input in text_inputs if input['label'] != filter_tag}
        st.success('Request submmited! : {}'.format(dict_iputs))

        prediction = ModelPredict() # pass the into main(dict_iputs)
        
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

