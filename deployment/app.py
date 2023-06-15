import pandas as pd
import numpy as np
import streamlit as st
# Library Load Model
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Library Pre-Processing
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec

# Load tokenizer
with open("tokenizer.pkl", "rb") as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

#load model
model_rnn = load_model('model_improve.h5')
# build text cleaning function
def clean_text(x):
    '''
    Clean the text data by applying various operations to input text.
    Parameters:
        text (str): The input text to be cleaned.
    Returns:
        str: The cleaned text.
        '''
    text = x
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub('\n', '', text)
    return text
def remove_stopwords(text):
    '''
    Function to remove stopwords from text using NLTK library.
    Parameters:
        text (str): The input text to be cleaned.
    Returns:
        str: The cleaned text with stopwords removed.
    '''
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    cleaned_text = ' '.join(filtered_tokens)
    return cleaned_text
def lemmatize_text(text):
    '''
    Function to lemmatize text using NLTK library.
    Parameters:
        text (str): The input text to be lemmatized.
    Returns:
        str: The lemmatized text.
    '''
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    lemmatized_text = ' '.join(lemmatized_tokens)
    return lemmatized_text
def preprocess_text(text):
    '''
    Function to preprocess text by cleaning, removing stopwords, and lemmatizing.

    Parameters:
        text (str): The input text to be preprocessed.

    Returns:
        str: The preprocessed text.
    '''
    text = clean_text(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text

def run():
    # membuat title
    st.title('FAKE NEWS DETECTION')
    st.subheader('Detecting Fake News')
    st.markdown('---')
    # Buat form
    with st.form(key='fake_news_detect'):
        st.write("## News text")
        # URL input
        text = st.text_input("Enter the news article main text:")
        submitted = st.form_submit_button('Predict')
        # Perform prediction
        if submitted:
                data_inf = {'text': text}
                data_inf = pd.DataFrame([data_inf])
                # Preprocess the text (apply the same preprocessing steps as used during training)
                data_inf['text'] = data_inf['text'].apply(lambda x: preprocess_text(x))
                data_inf = tokenizer.texts_to_sequences(data_inf)
                data_inf = pad_sequences(data_inf, maxlen=700)
                # Make the prediction using the loaded model
                y_pred_inf = model_rnn.predict(data_inf)
                y_pred_inf = np.where(y_pred_inf >= 0.5, 1, 0)

                # Display the prediction result
                if y_pred_inf == 0:
                    st.subheader("Prediction: Real News")
                else:
                    st.subheader("Prediction: Fake News")

                # Display the extracted text
                st.subheader("Extracted Text:")
                st.write(text)

if __name__ == '__main__':
    run()