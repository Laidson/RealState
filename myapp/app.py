# start with streamlit |terminal: streamlit run myapp/app.py |https://medium.com/geekculture/how-to-run-your-streamlit-apps-in-vscode-3417da669fc
#emoji https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/
import os
import sys
import streamlit as st 
import pandas as pd
from src.metrics import MetricCalulation

#sys.path.append(os.path.normpath(os.getcwd() + "/src"))

st.title(':house_with_garden: Real State Machine Learning app :robot_face:')
st.markdown('The purpose of this app is to provide summary stats :bar_chart: based on ML data search.')
st.markdown('#### {0} :point_down:'.format('Upload a csv file'))

uploaded_file = st.file_uploader('Choose a file')
if uploaded_file is not None:
    #read csv
    df = pd.reade_csv(uploaded_file)

#TODO retirar df abixo apenas para fixar o df de base de desenvolvimento
df = pd.read_csv('data/redfin_2023.csv')
#st.write(df.head()) # write first 5 rows (remove after testing)

# METRICS
calcmetric = MetricCalulation()

st.markdown('## Property Metrics :memo:')
col1,col2,col3,col4 = st.columns(4)
col1.metric('Total', calcmetric.num_of_propeties(df), help='Number of properties in search')
col2.metric('Avg Price',calcmetric.avg_price(df), help='Average sale price of properties in search')
col3.metric('Avg DOM', calcmetric.avg_dom(df), help='Average days on market of properties in search')
col4.metric('Avg PPSQFT', calcmetric.avg_ppsqft(df), help='Average price per square foot of properties in search')

# CHARTS