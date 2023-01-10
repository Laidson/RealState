# start with streamlit |terminal: streamlit run myapp/app.py |https://medium.com/geekculture/how-to-run-your-streamlit-apps-in-vscode-3417da669fc
#emoji https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/
import os
import sys
import streamlit as st 
import pandas as pd
from src.etl import DataManipulation
from src.charts import Charts
from src.newfeatures import NewFeatures
from src.metrics import MetricCalulation


#sys.path.append(os.path.normpath(os.getcwd() + "/src"))

st.title(':house_with_garden: Real State Machine Learning app :robot_face:')
st.markdown('The purpose of this app is to provide summary stats :bar_chart: based on ML data search.')
st.markdown('#### {0} :point_down:'.format('Upload a csv file'))

uploaded_file = st.file_uploader('Choose a file')
if uploaded_file is not None:
    #read csv
    df = pd.read_csv(uploaded_file)
    df = DataManipulation().convert_df(df)
    df.to_csv('data/file.csv')
    #uploaded_file.to_csv('data/file.csv')
    

#TODO retirar df abixo apenas para fixar o df de base de desenvolvimento
#df = pd.read_csv('data/redfin_2023.csv')
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
with st.expander('Charts :bar_chart:', expanded=True):
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
with st.expander('Oportunities :dizzy:', expanded=True):
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
with st.expander('ML forecasting :robot_face:'):
    st.markdown('## ML forecasting :robot_face:')

    #forecasting dataset
    df_prediction = pd.DataFrame()

    # convert forecasting dataset to csv
    csv = etl.convert_df(df_prediction)
    st.download_button(
        "Download 🔽",
        csv,
        "property_dataset.csv",
        "text/csv",
        key='download-csv'
    )

