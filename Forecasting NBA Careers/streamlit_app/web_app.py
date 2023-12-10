import streamlit as st
from PIL import Image

st.set_page_config(
page_title="Welcome",
page_icon=":basketball:",
layout="wide",
initial_sidebar_state="expanded")

#The title
st.title("NBA Analytics :basketball:")

#The subheader
st.subheader("Revolutionize the game through data")

#The text
st.write("Introduction in process...")

image = Image.open('revenues_ev.png')

st.image(image, caption='Revenues evolution through the years')


