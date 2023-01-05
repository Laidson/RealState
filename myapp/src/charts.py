import plotly.express as px
import streamlit as st

class Charts:
    def __init__(self) -> None:
        pass
    
    def create_fig_histogram(self, df, column_name, title):
        fig=px.histogram(df, x=column_name, title=title)
        return st.plotly_chart(fig, use_container_width=True)
    
    def create_fig_boxplot(self, df, column_name, title):
        fig = px.box(df, x=column_name, title=title)
        return st.plotly_chart(fig, use_container_width=True)
         