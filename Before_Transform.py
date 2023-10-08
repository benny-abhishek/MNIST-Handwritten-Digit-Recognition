import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import accuracy_score
import pickle

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
    st.title(":red[MNIST Handwritten Digit (Before Transformation)]")

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

    #resize for algos
    x_train = x_train.reshape(x_train.shape[0], 28*28) / 255.0
    x_test = x_test.reshape(x_test.shape[0], 28*28) / 255.0

    #SVM
    from sklearn.svm import SVC
    x_train_subset = x_train[:10000]
    y_train_subset = y_train[:10000]

    svm = SVC(kernel='linear')
    svm.fit(x_train_subset, y_train_subset)
    svm_pred = svm.predict(x_test)

    with open('svm_model.pkl', 'wb') as file:
        # Serialize and save the model to the file
        pickle.dump(svm, file)

    # Close the file
    file.close()

    st.subheader(":violet[Support Vector Machine (SVM)]")

    plt.style.use('dark_background')
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(1.5*cols, 2*rows))
    for i, ax in enumerate(axes.flatten()):
        image = x_test[i].reshape(28, 28)    # returns PIL image with its labels
        label=svm_pred[i]
        ax.axis('off')
        ax.set_title(f"Predicted: {label}")
        ax.imshow(image, cmap='gray')  # we get a 1x28x28 tensor -> remove first dimension
    st.pyplot(fig)
    svm_acc = accuracy_score(y_test, svm_pred)
    st.subheader(":violet[The accuracy score is: ]"+str(svm_acc))
    st.markdown("""---""")

    #Kneibours
    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(x_train, y_train)
    knn_pred = knn.predict(x_test)

    with open('knn_model.pkl', 'wb') as file:
        # Serialize and save the model to the file
        pickle.dump(knn, file)

    # Close the file
    file.close()

    st.header(":blue[K-Neighbours Classifier]")

    plt.style.use('dark_background')
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(1.5*cols, 2*rows))
    for i, ax in enumerate(axes.flatten()):
        image = x_test[i].reshape(28, 28)    # returns PIL image with its labels
        label=knn_pred[i]
        ax.axis('off')
        ax.set_title(f"Predicted: {label}")
        ax.imshow(image, cmap='gray')  # we get a 1x28x28 tensor -> remove first dimension
    st.pyplot(fig)
    knn_acc = accuracy_score(y_test, knn_pred)
    st.subheader(":blue[The accuracy score is: ]"+str(knn_acc))
    st.markdown("""---""")

    #logisicreg
    from sklearn.linear_model import LogisticRegression

    logreg = LogisticRegression(max_iter=10)
    logreg.fit(x_train, y_train)
    logreg_pred = logreg.predict(x_test)

    with open('logreg_model.pkl', 'wb') as file:
        # Serialize and save the model to the file
        pickle.dump(logreg, file)

    # Close the file
    file.close()

    st.header(":green[Logisitc Regression]")

    plt.style.use('dark_background')
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(1.5*cols, 2*rows))
    for i, ax in enumerate(axes.flatten()):
        image = x_test[i].reshape(28, 28)    # returns PIL image with its labels
        label=logreg_pred[i]
        ax.axis('off')
        ax.set_title(f"Predicted: {label}")
        ax.imshow(image, cmap='gray')  # we get a 1x28x28 tensor -> remove first dimension
    st.pyplot(fig)
    logreg_acc = accuracy_score(y_test, logreg_pred)
    st.subheader(":green[The accuracy score is: ]"+str(logreg_acc))
    st.markdown("""---""")

    #MLP
    from sklearn.neural_network import MLPClassifier

    mlp = MLPClassifier(hidden_layer_sizes=(100,50), max_iter=10)
    mlp.fit(x_train, y_train)
    mlp_pred = mlp.predict(x_test)

    with open('mlp_model.pkl', 'wb') as file:
        # Serialize and save the model to the file
        pickle.dump(mlp, file)

    # Close the file
    file.close()

    st.header(":orange[Multi-Layer Perceptron]")

    plt.style.use('dark_background')
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(1.5*cols, 2*rows))
    for i, ax in enumerate(axes.flatten()):
        image = x_test[i].reshape(28, 28)    # returns PIL image with its labels
        label=mlp_pred[i]
        ax.axis('off')
        ax.set_title(f"Predicted: {label}")
        ax.imshow(image, cmap='gray')  # we get a 1x28x28 tensor -> remove first dimension
    st.pyplot(fig)
    mlp_acc = accuracy_score(y_test, mlp_pred)
    st.subheader(":orange[The accuracy score is: ]"+str(mlp_acc))
    st.markdown("""---""")

    import scipy.stats as stats
    # Perform one-way ANOVA
    f_val, p_val = stats.f_oneway(svm_pred == y_test, knn_pred == y_test, logreg_pred == y_test, mlp_pred == y_test)

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
            t_val, p_val = stats.ttest_ind(eval(model1.lower()+'_pred') == y_test, eval(model2.lower()+'_pred') == y_test)
            #print(f"t-test between {model1} and {model2}: t = {t_val}, p = {p_val}")
            t=[f"{model1} and {model2}",t_val,p_val]
            temp.append(t)
    #print(temp)

    df=pd.DataFrame(temp,columns=['Models','T-value','F-value'])
    st.table(df)
    



app()
