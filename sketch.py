import gradio as gr
import tensorflow as tf
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28*28) / 255.0
x_test = x_test.reshape(x_test.shape[0], 28*28) / 255.0

def predict(img):
    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(x_train, y_train)
    img_array = np.array(img)
    img_array = img_array.reshape(1,-1)
    img_array = img_array/255
    knn_pred = knn.predict(img_array)
    print(knn_pred[0])
    return knn_pred[0]

iface = gr.Interface(predict, inputs = 'sketchpad',
                     outputs = 'text',
                     allow_flagging = 'never',
                     description = 'Draw a Digit Below (Draw in the centre for best results)')
iface.launch(share = True, width = 300, height = 500)





