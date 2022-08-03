import seaborn as sns
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils import to_categorical
import time
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from keras.utils import plot_model

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

def loadData():
    data = pd.read_csv("C:/Users/krarv/Desktop/ProjectData/project16Data.csv",header=None,delimiter=',')
    x = data.drop(len(data.columns)-1, axis=1)
    y = data[len(data.columns)-1]
    x = np.array(x[:])
    y = np.array(y[:])
    y = to_categorical(y)
    xTrain, xTest, yTrain, yTest = train_test_split(x,y,test_size=0.3,random_state=8888,shuffle=True)
    xTrain = xTrain.reshape(xTrain.shape[0],xTrain.shape[1],1) #reshape to fit to CNN
    xTest = xTest.reshape(xTest.shape[0],xTest.shape[1],1) #reshape to fit to CNN
    return xTrain, xTest, yTrain, yTest

def CNN(xTrain, xTest, yTrain, yTest):
    input_shape = (xTrain.shape[1],1)
    model = keras.Sequential()
    model.add(layers.Conv1D(filters=16, kernel_size=4, activation='relu', input_shape=(input_shape),padding='same'))
    model.summary()
    model.add(layers.Flatten())
    model.add(layers.Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    model.summary()
    plot_model(model, to_file='1DCNN_Model.png', show_shapes=True, show_layer_names=True)
    epochs = 100
    start = time.time()
    history = model.fit(xTrain,yTrain, epochs=epochs,validation_data=(xTest,yTest))
    stop = time.time()
    print(f"Training time:{stop - start}s")
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('1D CNN model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    predictions = np.argmax(model.predict(xTest), axis=-1)
    model_json = model.to_json()
    with open("1DCNN_model","w") as json_file: #save model
        json_file.write(model_json)
    model.save_weights("1Dmodel.h5")
    return predictions


def showresults(predictions,yTest):
    yTest = np.argmax(yTest, axis=-1)
    confusion_matrix = metrics.confusion_matrix(yTest,predictions)
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    categories = ['Healthy','Cancerous']
    group_counts = ["{0:0.0f}".format(value) for value in confusion_matrix.flatten()]
    labels = [f"{v1}\n{v2}\n" for v1, v2, in zip(group_names,group_counts)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(confusion_matrix,annot=labels,fmt='',cmap='Blues',xticklabels=['healthy','cancerous'],yticklabels=['healthy','cancerous'])
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("AUC=" +  str(metrics.roc_auc_score(yTest,predictions)))
    plt.show()
    print("AUC:",metrics.roc_auc_score(yTest, predictions))


if __name__ == '__main__':
    xTrain, xTest, yTrain, yTest = loadData()
    predictions = CNN(xTrain, xTest, yTrain, yTest)
    showresults(predictions,yTest)
