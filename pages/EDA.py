import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import accuracy_score
import pickle
import numpy as np

def app():
    st.set_page_config(layout="wide")
    mainpage_bg = '''<style>
    [data-testid="stAppViewContainer"]>.main{{
    background-image:url("image/img_file.jpg");
    background-size : cover;
    background-position : top left;
    background-repeat : no-repeat;
    backgorund-attachment:local;}}
    [data-testid="stHeader"]
    {{background:rgba(0,0,0,0);
    }}
    [data-testid="stToolbar"]
    {{right: 2rem;}}
    </style>'''
    st.markdown(mainpage_bg,unsafe_allow_html=True)
    #Title
    st.title(":red[Exploratory Data Analysis]")

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    st.header("Sample Dataset")

    cols = 8
    rows = 2

    plt.style.use('dark_background')
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(1.5*cols, 2*rows))
    for i, ax in enumerate(axes.flatten()):
        image= x_train[i]           # returns PIL image with its labels
        label=y_train[i]
        ax.axis('off')
        ax.set_title(f"Label: {label}")
        ax.imshow(image, cmap='gray')  # we get a 1x28x28 tensor -> remove first dimension
    
    st.pyplot(fig)
    st.markdown("""---""")

    #Class Distribution
    st.header(":blue[Class Distirubution Plot]")

    colors = ['#E57373', '#FFD54F', '#81C784', '#64B5F6', '#FF8A65',
                 '#9575CD', '#4FC3F7', '#FFD180', '#AED581', '#FF80AB']
    unique, counts = np.unique(y_train, return_counts=True)
    fig=plt.figure(figsize=(10,4))
    plt.bar(unique, counts,color=colors)
    plt.xlabel("Digit Class")
    plt.ylabel("Count")
    plt.title("Class Distribution in Training Data")
    st.pyplot(fig)
    st.markdown("""---""")

    #pixel intensity
    st.header(":green[Pixel Intensity Distribution Graph]")
    fig=plt.figure(figsize=(10,4))
    plt.hist(x_train[0].ravel(), bins=256, range=(0, 255))
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.title("Pixel Intensity Distribution for a Sample Image")
    st.pyplot(fig)
    st.markdown("""---""")


app()

