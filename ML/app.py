from flask import Flask, render_template, request
import pickle
import numpy as np
from scipy.sparse import issparse

app = Flask(__name__)

# Load the pickled model tuple
with open('keyword.pkl', 'rb') as model_file:
    cv, tfidf_transformer = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form['a']

    # Convert to a list of strings
    arr = [data]

    # Transform input data using the loaded model components
    X_input = cv.transform(arr)
    X_tfidf = tfidf_transformer.transform(X_input)

    # Convert sparse matrix to dense array if needed
    if issparse(X_tfidf):
        X_tfidf = X_tfidf.toarray()

    # Get feature names from CountVectorizer
    feature_names = cv.get_feature_names_out()

    # Extract and display the top keywords
    top_keywords_indices = np.argsort(X_tfidf[0])[::-1][:5]  # Get indices of top 5 keywords
    top_keywords = [feature_names[idx] for idx in top_keywords_indices]

    # Print or log the top keywords for debugging
    print("Top Keywords:", top_keywords)

    return render_template('prediction.html', data=top_keywords)

if __name__ == '__main__':
    app.run(debug=True)
