from operator import index
import streamlit as st
import plotly.express as px
from pycaret.regression import setup, compare_models, pull, save_model, load_model
import pandas_profiling
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os

if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar: 
    st.title("📊 Data Analytics Tool")
    choice = st.radio("Navigation", ["Home","Data Upload","Profiling","Visualisation","Prediction"])
    st.markdown("👩‍💻 Connect with me on [LinkedIn](https://www.linkedin.com/in/lokepak-yen/)")
    st.info("💡 This application helps you explore your data using basic data analysis, visualisation and AI/ML predictive modelling.")

#Home
if choice == "Home":
    st.title("👋 Data Analytics Tool")
    st.info("[UPDATE] added download button - export your data as html")
    st.markdown("This is a very basic analytics web app project made using Streamlit. Through this app you will be able to do simple data manipulation, analysis, visualisation and automatic machine learning using regression models. Here are some basic information 👉")
    st.markdown("##### Data Preprocessing")
    st.markdown("First, upload your dataset (CSV file) and explore your data through the data viewer. If you have missing values, you can choose to keep, drop, fill or impute missing values. If you only want to examine a few column, deselect the columns you'd like to remove from this analysis. Once you're satisfied with the data, just head onto either tab - your data will be stored throughout the session.")
    st.markdown("##### Data Profile Report")
    st.markdown("This tool uses pandas_profiling package to return a data profile report based on your dataset.")
    st.markdown("##### Data Visualisation")
    st.markdown("This tool allows you to do basic visualisations by defining your target and entity variable.")
    st.markdown("##### Machine Learning Models (Regression)")
    st.markdown("This tool uses pycaret to automatically run a series of regression model based on your dataset and return their performance.")

#Data Upload and Preprocessing
if choice == "Data Upload":
    st.title("🔧Data Preprocessing")
    st.info("📍 [START HERE] Upload and clean your data here to use it in the following analysis. Once you have chosen your dataset, proceed onto the next tab. Only one dataset can be analysed per analysis.")
    file = st.file_uploader("Upload Your Dataset (ONLY CSV)")
    if file: 
        df = pd.read_csv(file, index_col=None)
        option = st.selectbox("Handle Missing Values", ["Keep Missing Values","Drop Missing Rows", "Fill Missing Values", "Impute Missing Values"])
        
        if option == "Drop Missing Rows":
            df = df.dropna()
        elif option == "Fill Missing Values":
            fill_value = st.text_input("Fill Value")
            df = df.fillna(fill_value)
        elif option == "Impute Missing Values":
            impute_method = st.radio("Imputation Method", ["Mean", "Median", "Mode"])
            impute_columns = st.multiselect("Columns to Impute", df.columns)

            if impute_method == "Mean":
                for column in impute_columns:
                    df[column] = df[column].fillna(df[column].mean())
            elif impute_method == "Median":
                for column in impute_columns:
                    df[column] = df[column].fillna(df[column].median())
            elif impute_method == "Mode":
                for column in impute_columns:
                    mode_value = df[column].mode().iloc[0]
                    df[column] = df[column].fillna(mode_value)

  
        drop_column = st.multiselect("Deselect Columns [Optional]", df.columns)
        df = df.drop(drop_column, axis=1)

        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

#Data Visualisation 
if choice == "Visualisation":
    st.title("🖥️ Data Visualisation")
    st.info("✏️ This tool generates basic visualisations with your chosen variables/columns.")
    graph_type = st.selectbox('Choose Graph', ['Scatter Plot', 'Bar Chart', 'Line Plot','Histogram','Heatmap','Scatter Matrix'])
    chosen_target = st.selectbox('Choose Target', df.columns)
    chosen_entity = st.selectbox('Choose Entity', df.columns)

    if graph_type == 'Scatter Plot':
        fig = px.scatter(df, x=chosen_entity, y=chosen_target)
    elif graph_type == 'Histogram':
        fig = px.histogram(df, x=chosen_entity, y=chosen_target)
    elif graph_type == 'Heatmap':
        fig = px.imshow(df)
    elif graph_type == 'Bar Chart':
        fig = px.bar(df, x=chosen_entity, y=chosen_target)
    elif graph_type == 'Line Plot':
        fig = px.line(df, x=chosen_entity, y=chosen_target)
    elif graph_type == 'Scatter Matrix':
        fig = px.scatter_matrix(df)  

    st.plotly_chart(fig)

#Data Profile Report
if choice == "Profiling": 
    st.title("🐼 Data Profile Report")
    st.info("✏️ This tool uses pandas_profiling to produce a general profile of your current dataset.")
    profile_df = df.profile_report()
    st_profile_report(profile_df)
    export_profile = profile_df.to_html()
    st.download_button(label='Download Data Profile', data=export_profile, file_name='profile_report.html')

#Prediction
if choice == "Prediction": 
    st.title("🔮 Machine Learning Models (Regression)")
    st.info("✏️ This tool will run a series of regression models using pycaret on your chosen target variable and return model performance. Remember to preprocess your data before running the models!")
    chosen_target = st.selectbox('Choose Target Column', df.columns)
    if st.button('Run Modelling'): 
        setup(df, target=chosen_target)
        setup_df = pull()
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        save_model(best_model, 'best_model')
        export_prediction = compare_df.to_html()
        st.download_button(label='Download Data Profile', data=export_prediction, file_name='prediction.html')
