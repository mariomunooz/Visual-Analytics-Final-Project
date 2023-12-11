import pickle
import pandas as pd
import re
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import streamlit as st
from PIL import Image
import requests
from io import BytesIO


def display_image_from_url(player_data_df):
    pl_nbapersonid = player_data_df['nbapersonid'].iloc[0]
    try:
        response = requests.get(f"https://cdn.nba.com/headshots/nba/latest/1040x760/{pl_nbapersonid}.png")
        img = Image.open(BytesIO(response.content))
        st.image(img, caption= f"{player_data_df['player'].iloc[0]}", width=300)
    except Exception as e:
        response = requests.get(f"https://cdn.nba.com/headshots/nba/latest/1040x760/fallback.png")
        img = Image.open(BytesIO(response.content))
        st.image(img, caption=f"{player_data_df['player'].iloc[0]} (No image available)", width=300)


def plot_boxplots_from_df(df, columns, title_txt):
    num_columns = len(columns)
    num_rows = -(-num_columns // 3)  # Calculate number of rows needed for 3 plots per row
    fig_height = num_rows * 300

    fig = make_subplots(rows=num_rows, cols=3, subplot_titles=columns)

    for i, col in enumerate(columns, start=1):
        row = ((i - 1) // 3) + 1
        col_in_row = ((i - 1) % 3) + 1

        boxplot = go.Box(y=df[col], name=f'{col} Boxplot')
        fig.add_trace(boxplot, row=row, col=col_in_row)

    fig.update_layout(title=title_txt, showlegend=True, height=fig_height)
    st.plotly_chart(fig)


def plot_distributions_from_df(df, columns, title_txt):
    num_columns = len(columns)
    num_rows = -(-num_columns // 3)
    fig_height = num_rows * 300

    fig = make_subplots(rows=num_rows, cols=3, subplot_titles=columns, horizontal_spacing=0.12)

    for i, col in enumerate(columns, start=1):
        row = ((i - 1) // 3) + 1
        col_in_row = ((i - 1) % 3) + 1

        histogram = go.Histogram(x=df[col], name=f'{col} Distribution')
        fig.add_trace(histogram, row=row, col=col_in_row)

        fig.update_xaxes(title_text=col, row=row, col=col_in_row)
        fig.update_yaxes(title_text='Count', row=row, col=col_in_row)

    fig.update_layout(title=title_txt, showlegend=True, height=fig_height)
    st.plotly_chart(fig)


def plot_distributions_by_target(df, column, target_column):
    unique_values = df[target_column].unique()
    num_unique_values = len(unique_values)

    # Convert unique values to strings
    unique_values_str = [str(value) for value in unique_values]

    # Calculate number of rows and columns
    num_rows = -(-num_unique_values // 3)  # Round up division to get the number of rows
    num_columns = min(num_unique_values, 3)  # Maximum 3 columns per row
    

    
    fig = make_subplots(rows=num_rows, cols=num_columns, subplot_titles=unique_values_str, horizontal_spacing=0.12)

    for i, value in enumerate(unique_values, start=1):
        row = ((i - 1) // num_columns) + 1  # Calculate the current row for subplot
        col = ((i - 1) % num_columns) + 1  # Calculate the column in the current row for subplot

        data = df[df[target_column] == value][column]
        histogram = go.Histogram(x=data, name=f'{column} for {value}', histnorm='probability density')
        fig.add_trace(histogram, row=row, col=col)

        fig.update_xaxes(title_text=column, row=row, col=col)
        fig.update_yaxes(title_text='Density', row=row, col=col)

    fig.update_layout(title=f'Distribution of {column} by {target_column}', showlegend=True)
    st.plotly_chart(fig)


def plot_boxplots_and_distributions_from_df(df, columns, title_txt):
    plot_boxplots_from_df(df, columns, title_txt)
    plot_distributions_from_df(df, columns, title_txt)


with open("dataframe_all.pkl","rb") as file:
    df = pickle.load(file)["dataframe"]


# Define variable groups
boxplot_vars = {
    'Time on Court': ['games', 'games_start', 'mins'],
    'Info': ['draftyear', 'draftpick', 'season', 'season_num', 'num_teams_played'],
    'Shooting Stats': ['fgm', 'fga', 'fgp', 'fgm2', 'fga2', 'fgp2', 'fgm3', 'fga3',
                       'fgp3', 'ftm', 'fta', 'ftp', 'efg', 'points'],
    'Rebounding': ['off_reb', 'def_reb', 'tot_reb'],
    'Passing': ['ast', 'tov'],
    'Defense': ['steals', 'blocks', 'tot_fouls']
}

#The title and text
st.title("League data Exploration 📊 ")
st.write("In this tab we can see the most relevant information that we can extract through the data from the visual analytics.")

# Unpack the boxplot_vars into separate variables to be able to call the functions at the top
time_on_court, info, shooting_stats, rebounding, passing, defense = boxplot_vars.values()

# Create a selectbox to choose which variable group to show, in this way we avoid showing big loads of graphs
selected_group = st.selectbox('Select the topic of the data you want to observe:',
                              options=list(boxplot_vars.keys()))

selected_variables = boxplot_vars[selected_group]

selected_options = st.multiselect('Select options for boxplot', options=selected_variables)


plot_boxplots_and_distributions_from_df(df, selected_options, f'Boxplot of the stats related to {selected_group}')



for variable in selected_options:
    #st.write(variable)
    plot_distributions_by_target(df, variable, 'career_outcome')


if st.button("Show league leaders:"):
    for variable in selected_options:
        idx_max = df[variable].idxmax()
        leader = df.iloc[idx_max]["player"]
        leader_val = df.iloc[idx_max][variable]
        leader_data = df[df["player"] == leader]
        display_image_from_url(leader_data)
        st.write(f"The all-time leader is {leader} with {leader_val} {variable}")
