import numpy as np
import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pickle

# Set the NLTK data directory if it's not set
nltk.data.path.append('/Users/chandraprakashbathula/nltk_data')

# Download the WordNet resource
nltk.download('wordnet')

# Set the input directory as the current working directory
input_directory = os.getcwd()

# Read the dataset from the CSV file
file_path = os.path.join(input_directory, 'papers.csv')
dataset = pd.read_csv(file_path)

# Perform text preprocessing
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
custom_stopwords = ["using", "show", "result", "large", "also", "iv", "one", "two", "new", "previously", "shown"]
stop_words.update(custom_stopwords)

lem = WordNetLemmatizer()

corpus = []
for i in range(len(dataset)):
    text = re.sub('[^a-zA-Z]', ' ', str(dataset['title'][i]))  # Convert to string
    text = text.lower()
    text = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text)
    text = re.sub("(\\d|\\W)+", " ", text)
    text = text.split()
    text = [lem.lemmatize(word) for word in text if word not in stop_words]
    text = " ".join(text)
    corpus.append(text)

# Create a CountVectorizer instance
cv = CountVectorizer(
    max_df=0.8,
    stop_words=None,  # Use the default English stopwords
    max_features=10000,
    ngram_range=(1, 3)
)

# Transform the 'corpus' into a document-term matrix
X = cv.fit_transform(corpus)

# Create a TfidfTransformer instance
tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)

# Fit the TF-IDF transformer to the CountVectorizer output 'X'
tfidf_transformer.fit(X)

# Save the trained model as a pickle file
with open('keyword.pkl', 'wb') as model_file:
    model = (cv, tfidf_transformer)
    pickle.dump(model, model_file)

print("Trained model saved as keyword.pkl.")
