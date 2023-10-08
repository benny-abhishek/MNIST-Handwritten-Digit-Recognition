import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

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

    #AFTER TRANSFORMATION
    
    st.title(":red[MNIST Handwritten Digit (After Brightness Dimmed Transformation)]")

    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    st.header("Sample Dataset")

    cols = 8
    rows = 2

    plt.style.use('dark_background')
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(1.5*cols, 2*rows))
    for i, ax in enumerate(axes.flatten()):
        image= X_train[i]           # returns PIL image with its labels
        label=y_train[i]
        ax.axis('off')
        ax.set_title(f"Label: {label}")
        ax.imshow(image, cmap='gray')  # we get a 1x28x28 tensor -> remove first dimension
    
    st.pyplot(fig)
    st.markdown("""---""")

    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    
    datagen = ImageDataGenerator(
        rotation_range=10, 
        zoom_range=0.1,
        width_shift_range=0.1, 
        height_shift_range=0.1,
        brightness_range=(0, 0.4)
    )

    st.subheader("Sample for Brightness Dimmed Dataset")
    it = datagen.flow(X_train, y_train, batch_size=9)
    batch = it.next()

    fig=plt.figure(figsize=(6,0.3))
    for i in range(8):
        plt.subplot(1, 8, i+1)
        plt.imshow(batch[0][i].reshape(28, 28), cmap='gray')
        plt.axis('off')
    st.pyplot(fig)
    st.markdown("""---""")

    augmented_X, augmented_y = it.next()
    X_train_augmented = np.vstack((X_train, augmented_X))
    y_train_augmented = np.hstack((y_train, augmented_y))

    X_train_augmented = X_train_augmented.reshape(X_train_augmented.shape[0], 28*28) / 255.0
    x_test =X_test.reshape(X_test.shape[0], 28*28) / 255.0

    #SVM
    from sklearn.svm import SVC

    x_train_subset = X_train_augmented[:10000]
    y_train_subset = y_train_augmented[:10000]

    svm1 = SVC(kernel='linear')
    svm1.fit(x_train_subset, y_train_subset)
    svm_pred1 = svm1.predict(x_test)

    st.subheader(":violet[Support Vector Machine (SVM)]")

    plt.style.use('dark_background')
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(1.5*cols, 2*rows))
    for i, ax in enumerate(axes.flatten()):
        image = x_test[i].reshape(28, 28)    # returns PIL image with its labels
        label=svm_pred1[i]
        ax.axis('off')
        ax.set_title(f"Predicted: {label}")
        ax.imshow(image, cmap='gray')  # we get a 1x28x28 tensor -> remove first dimension
    st.pyplot(fig)
    svm_acc1 = accuracy_score(y_test, svm_pred1)
    st.subheader(":violet[The accuracy score is: ]"+str(svm_acc1))
    st.markdown("""---""")

    #KNN
    from sklearn.neighbors import KNeighborsClassifier

    knn1 = KNeighborsClassifier(n_neighbors=3)
    knn1.fit(X_train_augmented, y_train_augmented)
    knn_pred1 = knn1.predict(x_test)

    st.header(":blue[K-Neighbours Classifier]")

    plt.style.use('dark_background')
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(1.5*cols, 2*rows))
    for i, ax in enumerate(axes.flatten()):
        image = x_test[i].reshape(28, 28)    # returns PIL image with its labels
        label=knn_pred1[i]
        ax.axis('off')
        ax.set_title(f"Predicted: {label}")
        ax.imshow(image, cmap='gray')  # we get a 1x28x28 tensor -> remove first dimension
    st.pyplot(fig)
    knn_acc1 = accuracy_score(y_test, knn_pred1)
    st.subheader(":blue[The accuracy score is: ]"+str(knn_acc1))
    st.markdown("""---""")

    #logisicreg
    from sklearn.linear_model import LogisticRegression

    logreg1 = LogisticRegression(max_iter=10)
    logreg1.fit(X_train_augmented, y_train_augmented)
    logreg_pred1 = logreg1.predict(x_test)

    st.header(":green[Logisitc Regression]")

    plt.style.use('dark_background')
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(1.5*cols, 2*rows))
    for i, ax in enumerate(axes.flatten()):
        image = x_test[i].reshape(28, 28)    # returns PIL image with its labels
        label=logreg_pred1[i]
        ax.axis('off')
        ax.set_title(f"Predicted: {label}")
        ax.imshow(image, cmap='gray')  # we get a 1x28x28 tensor -> remove first dimension
    st.pyplot(fig)
    logreg_acc1 = accuracy_score(y_test, logreg_pred1)
    st.subheader(":green[The accuracy score is : ]"+str(logreg_acc1))
    st.markdown("""---""")


    #MLPClassifier
    from sklearn.neural_network import MLPClassifier

    mlp1 = MLPClassifier(hidden_layer_sizes=(100,50), max_iter=10)
    mlp1.fit(X_train_augmented, y_train_augmented)
    mlp_pred1 = mlp1.predict(x_test)

    st.header(":orange[Multi-Layer Perceptron]")

    plt.style.use('dark_background')
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(1.5*cols, 2*rows))
    for i, ax in enumerate(axes.flatten()):
        image = x_test[i].reshape(28, 28)    # returns PIL image with its labels
        label=mlp_pred1[i]
        ax.axis('off')
        ax.set_title(f"Predicted: {label}")
        ax.imshow(image, cmap='gray')  # we get a 1x28x28 tensor -> remove first dimension
    st.pyplot(fig)
    mlp_acc1 = accuracy_score(y_test, mlp_pred1)
    st.subheader(":orange[The accuracy score is: ]"+str(mlp_acc1))
    st.markdown("""---""")

    import scipy.stats as stats
    # Perform one-way ANOVA
    f_val, p_val = stats.f_oneway(svm_pred1 == y_test, knn_pred1 == y_test, logreg_pred1 == y_test, mlp_pred1 == y_test)

    st.header(":red[One-way ANOVA]")
    st.subheader("F ="+ str(f_val))
    st.subheader("P-Value ="+ str(p_val))

    import pandas as pd
    # If p-value indicates a significant difference, perform pairwise t-tests
    temp=[]
    if p_val < 0.05:
        combinations = [('SVM', 'KNN'), ('SVM', 'LogReg'), ('SVM', 'MLP'), ('KNN', 'LogReg'), ('KNN', 'MLP'), ('LogReg', 'MLP')]
        for comb in combinations:
            model1, model2 = comb
            t_val, p_val = stats.ttest_ind(eval(model1.lower()+'_pred1') == y_test, eval(model2.lower()+'_pred1') == y_test)
            #print(f"t-test between {model1} and {model2}: t = {t_val}, p = {p_val}")
            t=[f"{model1} and {model2}",t_val,p_val]
            temp.append(t)
    #print(temp)

    df=pd.DataFrame(temp,columns=['Models','T-value','F-value'])
    st.table(df)









app()
