######################
# Import libraries
######################

import pandas as pd
import streamlit as st
import altair as alt
from PIL import Image
import keras
import tensorflow as tf
import numpy as np
from tools.utils import pipeline_inference, load_

######################
# Page Title
######################

st.title('🍿Movie Review Web Aplication')

######################
# Input Text Box
######################

#st.sidebar.header('Enter DNA sequence')
review = st.text_area('Enter the review', 'Write your own review ...', height=250)

############################
#  Load classifier 
############################
model, words = load_()

############################
#  Prediction
############################
thresh = 0.5
pred = pipeline_inference(review, words, model)

## Prints the input review text
st.header('MODEL INPUT')
st.write(review)

## Returns model output
st.header('MODEL OUTPUT (Sentiment)')
st.write('Positive review' if pred>thresh else 'Negative review')

