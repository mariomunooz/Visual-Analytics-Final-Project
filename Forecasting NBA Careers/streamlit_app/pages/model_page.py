import pickle
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder

class CustomLabelEncoder(LabelEncoder):
    def _init_(self, mapping):
        super()._init_()
        self.classes_ = list(mapping.keys())
        self.mapping = mapping

    def transform(self, x):
        return [self.mapping.get(item, -1) for item in x]
    


def create_label_encoder(mapping):
    return CustomLabelEncoder(mapping)




st.set_page_config(
page_title="Career Outcome Prediction",
layout="wide",
initial_sidebar_state="expanded")

# Read
with open('pages/model.pkl', 'rb') as file:
    data = pickle.load(file)
    
with open("dataframe_all.pkl","rb") as file:
    dataframe_all = pickle.load(file)["dataframe"]

model = data["model"]
le_career_outcome = data["le_career_outcome"]
le_season_outcome = data["le_season_outcome"]
X_test = data["X_test"]
y_test = data["y_test"]
y_pred_test = data["y_pred_test"]
young_players = data["young_players"]


#The title and text
st.title("Player career outcome prediction")
st.write("In this tab we can see the most relevant information that we can extract through the data from the visual analytics.")

columns_list = [
    'draftyear', 'draftpick', 'season', 'num_teams_played', 'games',
    'games_start', 'mins', 'fgm', 'fga', 'fgp', 'fgm2', 'fga2', 'fgp2',
    'fgm3', 'fga3', 'fgp3', 'ftm', 'fta', 'ftp', 'efg', 'off_reb',
    'def_reb', 'tot_reb', 'ast', 'steals', 'blocks', 'tov', 'tot_fouls',
    'points', 'season_outcome', 'season_num'
]

columns_list_per = [
    'fgp', 'fgp2', 'fgp3', 'ftp', 'efg'
]

draftyear  = st.slider("Draftyear", dataframe_all["draftyear"].min(), dataframe_all["draftyear"].max(), int(dataframe_all["draftyear"].mean()))

draftpick  = st.slider("Draftpick", -1, 60, 30)

# season  = st.slider("season", dataframe_all["season"].min(), dataframe_all["season"].max(), int(dataframe_all["season"].mean()))

name_variable_season = st.slider("season", dataframe_all["season"].min(), dataframe_all["season"].max(), int(dataframe_all["season"].mean()))
name_variable_num_teams_played = st.slider("num_teams_played", dataframe_all["num_teams_played"].min(), dataframe_all["num_teams_played"].max(), int(dataframe_all["num_teams_played"].mean()))
name_variable_games = st.slider("games", dataframe_all["games"].min(), dataframe_all["games"].max(), int(dataframe_all["games"].mean()))
name_variable_games_start = st.slider("games_start", dataframe_all["games_start"].min(), dataframe_all["games_start"].max(), int(dataframe_all["games_start"].mean()))
name_variable_mins = st.slider("mins", dataframe_all["mins"].min(), dataframe_all["mins"].max(), int(dataframe_all["mins"].mean()))
name_variable_fgm = st.slider("fgm", dataframe_all["fgm"].min(), dataframe_all["fgm"].max(), int(dataframe_all["fgm"].mean()))
name_variable_fga = st.slider("fga", dataframe_all["fga"].min(), dataframe_all["fga"].max(), int(dataframe_all["fga"].mean()))
name_variable_fgp = st.slider("fgp", dataframe_all["fgp"].min(), dataframe_all["fgp"].max(), (dataframe_all["fgp"].mean()))
name_variable_fgm2 = st.slider("fgm2", dataframe_all["fgm2"].min(), dataframe_all["fgm2"].max(), int(dataframe_all["fgm2"].mean()))
name_variable_fga2 = st.slider("fga2", dataframe_all["fga2"].min(), dataframe_all["fga2"].max(), int(dataframe_all["fga2"].mean()))
name_variable_fgp2 = st.slider("fgp2", dataframe_all["fgp2"].min(), dataframe_all["fgp2"].max(), (dataframe_all["fgp2"].mean()))
name_variable_fgm3 = st.slider("fgm3", dataframe_all["fgm3"].min(), dataframe_all["fgm3"].max(), int(dataframe_all["fgm3"].mean()))
name_variable_fga3 = st.slider("fga3", dataframe_all["fga3"].min(), dataframe_all["fga3"].max(), int(dataframe_all["fga3"].mean()))
name_variable_fgp3 = st.slider("fgp3", dataframe_all["fgp3"].min(), dataframe_all["fgp3"].max(), (dataframe_all["fgp3"].mean()))
name_variable_ftm = st.slider("ftm", dataframe_all["ftm"].min(), dataframe_all["ftm"].max(), int(dataframe_all["ftm"].mean()))
name_variable_fta = st.slider("fta", dataframe_all["fta"].min(), dataframe_all["fta"].max(), int(dataframe_all["fta"].mean()))
name_variable_ftp = st.slider("ftp", dataframe_all["ftp"].min(), dataframe_all["ftp"].max(), (dataframe_all["ftp"].mean()))
name_variable_efg = st.slider("efg", dataframe_all["efg"].min(), dataframe_all["efg"].max(), (dataframe_all["efg"].mean()))
name_variable_off_reb = st.slider("off_reb", dataframe_all["off_reb"].min(), dataframe_all["off_reb"].max(), int(dataframe_all["off_reb"].mean()))
name_variable_def_reb = st.slider("def_reb", dataframe_all["def_reb"].min(), dataframe_all["def_reb"].max(), int(dataframe_all["def_reb"].mean()))
name_variable_tot_reb = st.slider("tot_reb", dataframe_all["tot_reb"].min(), dataframe_all["tot_reb"].max(), int(dataframe_all["tot_reb"].mean()))
name_variable_ast = st.slider("ast", dataframe_all["ast"].min(), dataframe_all["ast"].max(), int(dataframe_all["ast"].mean()))
name_variable_steals = st.slider("steals", dataframe_all["steals"].min(), dataframe_all["steals"].max(), int(dataframe_all["steals"].mean()))
name_variable_blocks = st.slider("blocks", dataframe_all["blocks"].min(), dataframe_all["blocks"].max(), int(dataframe_all["blocks"].mean()))
name_variable_tov = st.slider("tov", dataframe_all["tov"].min(), dataframe_all["tov"].max(), int(dataframe_all["tov"].mean()))
name_variable_tot_fouls = st.slider("tot_fouls", dataframe_all["tot_fouls"].min(), dataframe_all["tot_fouls"].max(), int(dataframe_all["tot_fouls"].mean()))
name_variable_points = st.slider("points", dataframe_all["points"].min(), dataframe_all["points"].max(), int(dataframe_all["points"].mean()))
name_variable_season_outcome = st.slider("season_outcome", dataframe_all["season_outcome"].min(), dataframe_all["season_outcome"].max(), int(dataframe_all["season_outcome"].mean()))
name_variable_season_num = st.slider("season_num", dataframe_all["season_num"].min(), dataframe_all["season_num"].max(), int(dataframe_all["season_num"].mean()))

X_input = [draftyear,draftpick,name_variable_season,name_variable_num_teams_played, name_variable_games, name_variable_games_start, name_variable_mins, name_variable_fgm,
    name_variable_fga, name_variable_fgp, name_variable_fgm2, name_variable_fga2, name_variable_fgp2, name_variable_fgm3,
    name_variable_fga3, name_variable_fgp3, name_variable_ftm, name_variable_fta, name_variable_ftp, name_variable_efg,
    name_variable_off_reb, name_variable_def_reb, name_variable_tot_reb, name_variable_ast, name_variable_steals,
    name_variable_blocks, name_variable_tov, name_variable_tot_fouls, name_variable_points, name_variable_season_outcome, name_variable_season_num]




X_input_t = np.array(X_input).reshape(1, -1)
y_input = model.predict(X_input_t)

career_outcome_mapping = { 0: 'Out of the League', 1: 'Roster', 2: 'Rotation', 3: 'Starter', 4 : 'All-Star', 5 : 'Elite'}

st.write(career_outcome_mapping[y_input[0]])

