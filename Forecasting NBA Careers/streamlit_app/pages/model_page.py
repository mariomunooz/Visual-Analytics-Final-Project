import pickle
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

st.set_page_config(
page_title="Career Outcome Prediction",
layout="wide",
initial_sidebar_state="expanded")

#Load the pickle file with the model and the label encoders

with open('model.pkl', 'rb') as file:
    data = pickle.load(file)

model = data["model"]
le_career_outcome = data["le_career_outcome"]
le_season_outcome = data["le_season_outcome"]


