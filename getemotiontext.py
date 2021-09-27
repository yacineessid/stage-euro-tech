import streamlit as st 
import altair as alt
import plotly.express as px 

# EDA Pkgs
import pandas as pd 
import numpy as np 
from datetime import datetime
from textblob import TextBlob

from google_trans_new import google_translator
import matplotlib.pyplot as plt
import neattext.functions as nfx
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,plot_confusion_matrix

from sklearn.model_selection import train_test_split
import joblib
import os, glob, pickle
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,plot_confusion_matrix

from sklearn.model_selection import train_test_split

df=pd.read_csv("result/emotion_dataset_2.csv")
# Track Utils
nv_model = joblib.load("result/emotionClassifier.pkl")

#print(df.head())
df['Clean_Text'] = df ['Text'].apply(nfx.remove_stopwords)
df['Clean_Text'] = df ['Clean_Text'].apply(nfx.remove_userhandles)
df['Clean_Text'] = df ['Clean_Text'].apply(nfx.remove_punctuations)


def extract_keywords(text,num=50):
    tok =1
    tokens = [ tok for token in text.split()]
    most_common_tokens= Counter(tokens).most_common(num)
    return dict(most_common_tokens)



def predict_emotion(exmple_text2):


    trans = google_translator()
    translated = trans.translate(exmple_text2, lang_src = 'fr', lang_tgt = 'en')
    exmple_text =[translated]
    Xfeatures =df['Clean_Text']
    cv= CountVectorizer()
    X=cv.fit_transform(Xfeatures)
    cv.get_feature_names()
    vect = cv.transform(exmple_text).toarray()
    nv_model.predict(vect)
    nv_model.predict_proba(vect)
    nv_model.classes_
    return (nv_model.predict(vect),nv_model.predict_proba(vect),nv_model.classes_)
