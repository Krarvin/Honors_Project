import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree
import seaborn as sns
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt

def loadData():
    data = pd.read_csv("C:/Users/krarv/Desktop/ProjectData/project16Data.csv",header=None,delimiter=',')
    data.info()
    x = data.drop(len(data.columns)-1, axis=1)
    y = data[len(data.columns)-1]
    xTrain,xTest,yTrain,yTest = train_test_split(x,y,test_size = 0.3,random_state = 8,shuffle=True,stratify=y)
    return xTrain, xTest, yTrain, yTest

def decisionTree(xTrain, xTest, yTrain, yTest):
    dt = DecisionTreeClassifier()
    start = time.time()
    model = dt.fit(xTrain,yTrain)
    stop = time.time()
    print(f"Training time:{stop - start}s")
    predictions = model.predict(xTest)
    text_representation = tree.export_text(model, max_depth= 2)
    #fig = plt.figure(figsize=(10,10))
    #_ = tree.plot_tree(model, max_depth=2,fontsize=8)
    #fig.savefig("decision_tree.png")
    filename = 'dt_model.sav'
    pickle.dump(model, open(filename, 'wb'))
    return predictions

def showresults(predictions,yTest):
    confusion_matrix = metrics.confusion_matrix(yTest,predictions)
    print(confusion_matrix)
    print("AUC:",metrics.roc_auc_score(yTest, predictions))
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    categories = ['Healthy','Cancerous']
    group_counts = ["{0:0.0f}".format(value) for value in confusion_matrix.flatten()]
    labels = [f"{v1}\n{v2}\n" for v1, v2,in zip(group_names,group_counts)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(confusion_matrix,annot=labels,fmt='',cmap='Blues',xticklabels=['healthy','cancerous'],yticklabels=['healthy','cancerous'])
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Decision Tree Model.\nAUC=" +  str(metrics.roc_auc_score(yTest,predictions)))
    plt.show()

if __name__ == '__main__':
    xTrain, xTest, yTrain, yTest = loadData()
    predictions = decisionTree(xTrain, xTest, yTrain, yTest)
    showresults(predictions,yTest)

