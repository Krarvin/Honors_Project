from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
from sklearn.metrics import accuracy_score

def loadData():
    data = pd.read_csv("C:/Users/krarv/Desktop/ProjectData/project16Data.csv",header=None,delimiter=',')
    data.info()
    x = data.drop(len(data.columns)-1, axis=1)
    y = data[len(data.columns)-1]
    xTrain, xTest, yTrain, yTest = train_test_split(x,y,test_size=0.3,random_state=8888,shuffle=True)
    return xTrain, xTest, yTrain, yTest

def MLP(xTrain, xTest, yTrain, yTest):
    mlp = MLPClassifier(hidden_layer_sizes=(32,32,32), activation='relu', max_iter=100, solver='adam', verbose=True, tol=0.001)
    start = time.time()
    mlp.fit(xTrain,yTrain)
    stop = time.time()
    print(f"Training time:{stop - start}s")
    predictions = mlp.predict(xTest)
    filename = 'mlp_model.sav'
    pickle.dump(mlp, open(filename, 'wb'))
    return predictions

def showResults(yTest,predictions):
    print(classification_report(yTest,predictions))
    confusion_matrix = metrics.confusion_matrix(yTest,predictions)
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    categories = ['Healthy','Cancerous']
    group_counts = ["{0:0.0f}".format(value) for value in confusion_matrix.flatten()]
    labels = [f"{v1}\n{v2}\n" for v1, v2, in zip(group_names,group_counts)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(confusion_matrix,annot=labels,fmt='',cmap='Blues',xticklabels=['healthy','cancerous'],yticklabels=['healthy','cancerous'])
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("AUC=" +  str(metrics.accuracy_score(yTest,predictions)))
    plt.show()


if __name__ == '__main__':
    xTrain, xTest, yTrain, yTest = loadData()
    predictions = MLP(xTrain, xTest, yTrain, yTest)
    showResults(predictions,yTest)


