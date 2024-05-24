import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pickle
import os

# Importing the Necessary Libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import warnings

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 
from sklearn.tree import ExtraTreeRegressor
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

st.set_page_config(
    page_title="Industrial Copper Modeling",
    page_icon="random",
    layout="wide"
)

st.markdown ("WELCOME TO INDUSTRIAL COPPER MODELLING")

selected = option_menu(None, ['HOME',"PRICE PREDICTION","STATUS PREDICTION","CONCLUSION"])
    
if selected == "About Project":
    st.markdown("Industrial Copper Modeling")
    st.markdown("Technologies : Python, Pandas, Numpy, Scikit-Learn, Streamlit, Python scripting, "
                "Machine Learning, Data Preprocessing, Visualization, EDA, Model Building, Data Wrangling, "
                "Model Deployment")
    st.markdown("Overview : This project aims to construct a machine learning model and implement "
                "it as a user-friendly online application in order to provide accurate predictions about the "
                "Selling price and status of the copper model. This prediction model will be based on past transactions.")   


if selected == "Selling Price Prediction":
    st.markdown("Predicting Results based on Trained Model")
    a1 = st.text_input("Quantity")
    b1 = st.text_input("Status")
    c1 = st.text_input("Item Type")
    d1 = st.text_input("Application")
    e1 = st.text_input("Thickness")
    f1 = st.text_input("Width")
    g1 = st.text_input("Country")
    h1 = st.text_input("Customer")
    i1 = st.text_input("Product Reference")
            
    with open (r"C:\\Users\\The Evil King\\Desktop\\Reading Materila\\Assignments\\Industrial Copper\\model.pkl", 'rb') as file_1:
        regression_model = pickle.load(file_1)
 
    predict_button_1 = st.button("Predict Selling Price")

    if predict_button_1:

        a1 = float(a1)
        b1 = float(b1)
        c1 = float(c1)
        d1 = float(d1)
        e1 = float(e1)
        f1 = float(f1)
        g1 = float(g1)
        h1 = float(h1)
        i1 = float(i1)

        new_sample_1 = np.array(
                [[np.log(a1), b1, c1, d1, np.log(e1), f1, g1, h1, i1]])
        new_pred_1 = regression_model.predict(new_sample_1)[0]

        st.write('Predicted resale price:', np.exp(new_pred_1))

if selected == "Status (Win/Lost)":
    st.markdown("Predicting Results based on Trained Model")
    # -----New Data inputs from the user for predicting the status-----
    a2 = st.text_input("Quantity")
    b2 = st.text_input("Selling Price")
    c2 = st.text_input("Item Type")
    d2 = st.text_input("Application")
    e2 = st.text_input("Thickness")
    f2 = st.text_input("Width")
    g2 = st.text_input("Country")
    h2 = st.text_input("Customer")
    i2 = st.text_input("Product Reference")
            
    with open(r"C:\\Users\\The Evil King\\Desktop\\Reading Materila\\Assignments\\Industrial Copper\\classifer_model.pkl", 'rb') as file_2:
        classification_model = pickle.load(file_2)

  
    predict_button_2 = st.button("Predict Status")

    if predict_button_2:

        a2 = float(a2)
        b2 = float(b2)
        c2 = float(c2)
        d2 = float(d2)
        e2 = float(e2)
        f2 = float(f2)
        g2 = float(g2)
        h2 = float(h2)
        i2 = float(i2)

        new_sample_2 = np.array(
                [[np.log(a2), np.log(b2), c2, d2, np.log(e2), f2, g2, h2, i2]])
        new_pred_2 = classification_model.predict(new_sample_2)
            
        if new_pred_2 ==1:
            st.write('green[The Status is: Won]')
        else:
            st.write('green[The Status is: Lost]')