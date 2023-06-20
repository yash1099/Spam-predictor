import streamlit as st
import numpy as np
import pandas as pd
import re
import nltk
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('stopwords')
from nltk.corpus import stopwords
import string
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#Load the data 
df= pd.read_csv(r'C:\Users\Yash\Documents\spam_streamlit\spam.csv',encoding='latin-1')

#data Cleaning and preprocessing
df = df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
df.rename(columns={'v1': 'labels','v2': 'text'},inplace=True)
df.drop_duplicates(inplace=True)
# print(df.shape)
df['labels']= df['labels'].map({'ham':0,'spam': 1})
# print(df.head())

def clean_data(text):
    text_wo_punct= [character for character in text if character not in string.punctuation]
    text_wo_punct = ''.join(text_wo_punct)
    separator=' '
    return separator.join([word for word in text_wo_punct.split() if word.lower() not in stopwords.words('english')])

df['text']= df['text'].apply(clean_data)

#split the dataset for vectorizing
x=df['text']
y=df['labels']

#converting text into vectors
cv = CountVectorizer()

x= cv.fit_transform(x)
# print(x)

#train-test split the data
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.2,random_state=42)

#model 
model=MultinomialNB().fit(x_train,y_train)


#prdiction
predictions= model.predict(x_test)

# print(accuracy_score(y_test,predictions))
# print(classification_report(y_test,predictions))
# print(confusion_matrix(y_test,predictions) )

# function accepting text from user
def predict(text):
    labels=['Not Spam','Spam']
    x = cv.transform(text).toarray()
    p = model.predict(x)
    s = [str(i) for i in p]
    v= int(''.join(s))  # to convert the string into integer
    return str('This message seems like a:'+labels[v])

# print(predict(['congratulations you have won a lottery of $330000']))

st.title('Spam Classifier')
st.image('spam_img.jpg')
user_input=st.text_input('Write your message')
submit= st.button('Predict')
if submit:
    answer=predict([user_input])
    st.text(answer)

