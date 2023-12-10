import pickle
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.tri as tri

st.set_page_config(
page_title="Data Exploration",
page_icon="📊",
layout="wide",
initial_sidebar_state="expanded")

#The title and text
st.title("Data Exploration 📊 ")
st.write("In this tab we can see the most relevant information that we can extract through the data from the visual analytics.")

def hex_plot2(proportions1, proportions2, labels):
    def plot(ax, proportions, labels):
        N = len(proportions)
        proportions = np.append(proportions, 1)
        theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
        x = np.append(np.sin(theta), 0)
        y = np.append(np.cos(theta), 0)
        triangles = [[N, i, (i + 1) % N] for i in range(N)]
        triang_backgr = tri.Triangulation(x, y, triangles)
        triang_foregr = tri.Triangulation(x * proportions, y * proportions, triangles)
        cmap = plt.cm.rainbow_r
        colors = np.linspace(0, 1, N + 1)
        
        ax.tripcolor(triang_backgr, colors, cmap=cmap, shading='gouraud', alpha=0.4)
        ax.tripcolor(triang_foregr, colors, cmap=cmap, shading='gouraud', alpha=0.8)
        ax.triplot(triang_backgr, color='white', lw=2)
        for label, color, xi, yi in zip(labels, colors, x, y):
            ax.text(xi * 1.05, yi * 1.05, label,
                    ha='left' if xi > 0.1 else 'right' if xi < -0.1 else 'center',
                    va='bottom' if yi > 0.1 else 'top' if yi < -0.1 else 'center')
        ax.axis('off')
        ax.set_aspect('equal')
    
    # Create the plots side by side
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    plot(axs[0], proportions1, labels)
    plot(axs[1], proportions2, labels)
    
    st.pyplot(fig)  # Adjusts subplots to avoid overlapping
    
    return fig

# def hex_plot(proportions, labels):
#     N = len(proportions)
#     proportions = np.append(proportions, 1)
#     theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
#     x = np.append(np.sin(theta), 0)
#     y = np.append(np.cos(theta), 0)
#     triangles = [[N, i, (i + 1) % N] for i in range(N)]
#     triang_backgr = tri.Triangulation(x, y, triangles)
#     triang_foregr = tri.Triangulation(x * proportions, y * proportions, triangles)
#     cmap = plt.cm.rainbow_r  # or plt.cm.hsv ?
#     colors = np.linspace(0, 1, N + 1)
    
#     # Create the plot
#     fig, ax = plt.subplots()
#     ax.tripcolor(triang_backgr, colors, cmap=cmap, shading='gouraud', alpha=0.4)
#     ax.tripcolor(triang_foregr, colors, cmap=cmap, shading='gouraud', alpha=0.8)
#     ax.triplot(triang_backgr, color='white', lw=2)
#     for label, color, xi, yi in zip(labels, colors, x, y):
#         ax.text(xi * 1.05, yi * 1.05, label,
#                 ha='left' if xi > 0.1 else 'right' if xi < -0.1 else 'center',
#                 va='bottom' if yi > 0.1 else 'top' if yi < -0.1 else 'center')
#     ax.axis('off')
#     ax.set_aspect('equal')
    
#     # Display the plot in Streamlit
#     st.pyplot(fig)



with open("dataframe_all.pkl","rb") as file:
    dataframe_all = pickle.load(file)["dataframe"]

#List of all possible names
names = dataframe_all["player"].unique()

# Text input for user input
user_input = st.text_input('Type a player name:', '')

# Filter player names based on user input
filtered_players = [player for player in names if user_input.lower() in player.lower()]

# Display the filtered player names as options
selected_player = st.selectbox('Select a player:', options=filtered_players)

#['nbapersonid', 'player', 'draftyear', 'draftpick', 'career_outcome', 'season', 'team_list', 'num_teams_played', 'games', 'games_start','mins', 'fgm', 'fga', 'fgp', 
# 'fgm2', 'fga2', 'fgp2', 'fgm3', 'fga3','fgp3', 'ftm', 'fta', 'ftp', 'efg', 'off_reb', 'def_reb', 'tot_reb','ast', 'steals', 
# 'blocks', 'tov', 'tot_fouls', 'points','season_outcome', 'season_num']

player_data = dataframe_all[dataframe_all["player"]==selected_player]

#Career averages hex plot
games_pl = player_data["games"].sum()
ppg_career = player_data["points"].sum() / games_pl
reb_career = player_data["tot_reb"].sum() / games_pl
ast_career = player_data["ast"].sum() / games_pl
ste_career = player_data["steals"].sum() / games_pl
blo_career = player_data["blocks"].sum() / games_pl
tov_career = player_data["tov"].sum() / games_pl
player_av_stats = [ppg_career,reb_career,ast_career,ste_career,blo_career,tov_career]

league_max = [dataframe_all["points"].max()/82,dataframe_all["ast"].max()/82,dataframe_all["tot_reb"].max()/82,dataframe_all["steals"].max()/82,dataframe_all["blocks"].max()/82,dataframe_all["tov"].max()/82]
labels = ["ppg","astpg","rebpg","stpg","blpg","tov"]
proportions = [player_av_stats[i]/league_max[i] for i in range(len(league_max))]
#hex_plot2(proportions,labels)


#Career peak
ppg_peak_career = player_data["points"].max() / 82
reb_peak_career = player_data["tot_reb"].max() / 82
ast_peak_career = player_data["ast"].max() / 82
ste_peak_career = player_data["steals"].max() / 82
blo_peak_career = player_data["blocks"].max() / 82
tov_peak_career = player_data["tov"].max() / 82
player_peak_stats = [ppg_peak_career,reb_peak_career,ast_peak_career,ste_peak_career,blo_peak_career,tov_peak_career]

proportions_peak = [player_peak_stats[i]/league_max[i] for i in range(len(league_max))]
hex_plot2(proportions, proportions_peak,labels)



if st.button("Show season stats:"):
    st.write(player_data)
