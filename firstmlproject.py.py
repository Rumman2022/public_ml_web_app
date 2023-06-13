import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

dataset_names = st.sidebar.selectbox("Dataset Names",["IRIS","BREAST CANCER","WINE"])
classifire_names = st.sidebar.selectbox("Classifires",["KNN","RANDOM FOREST","SVM"])


st.header("MY FIRST PROJECT FOR MACHINE LEARNING IN STREAMLIT")




def load_datas(dataset_names):
    if dataset_names == "IRIS":
        data = datasets.load_iris()
    elif dataset_names == "BREAST CANCER":
        data = datasets.load_breast_cancer()
    else:

        data = datasets.load_wine()
    X = data.data
    y = data.target
    return X,y

X , y = load_datas(dataset_names)

st.write("The shape of data:", X.shape)
st.write("Number of classes:",len(np.unique(y)))


def parameter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K",1,10)
        params["K"] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C",0.01,10.0)
        params["C"] = C
    else:
        max_depth = st.sidebar.slider("max_depth",1,15)
        n_estimators = st.sidebar.slider("n_estimators",1,100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators

    return params

params = parameter_ui(classifire_names)

def get_classifier(clf_name , params):

    if clf_name == "KNN":
        clf =KNeighborsClassifier(n_neighbors=params["K"])


    elif clf_name == "SVM":
        clf = SVC(C=params["C"])

    else:
        clf = RandomForestClassifier(max_depth = params["max_depth"],
                                     n_estimators = params['n_estimators'],random_state = 123)


    return clf


clf = get_classifier(classifire_names,params)


xtrain, xtest,ytrain,ytest = train_test_split(X,y ,test_size= 0.2,random_state=21)

model = clf.fit(xtrain,ytrain)
ypred = model.predict(xtest)

acc = accuracy_score(ytest,ypred)

st.write(f"The model is {classifire_names}")
st.subheader(f"The accuracy score is {round(acc* 100)}")