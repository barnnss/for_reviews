from flask import Flask, request, jsonify
import pickle
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

app = Flask(__name__)


def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    text = word_tokenize(text)
    text = [word for word in text if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text]
    return ' '.join(text)


def calculate_cosine_similarity(review_vector, positive_vector, negative_vector):
    positive_similarity = cosine_similarity(
        review_vector, positive_vector)[0][0]
    negative_similarity = cosine_similarity(
        review_vector, negative_vector)[0][0]
    return positive_similarity, negative_similarity


current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'sentiment_model.pkl')
vectorizer_path = os.path.join(current_dir, 'vectorizer.pkl')


# Load model and vectorizer
try:
    with open('sentiment_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vec_file:
        vectorizer = pickle.load(vec_file)
except FileNotFoundError:
    print("Error: Model files not found in the specified directory.")
    exit(1)

# Define reference texts for positive and negative sentiments
positive_reference_text = "good excellent great amazing positive wonderful love perfect favorite like upgrade better best"
negative_reference_text = "bad terrible awful horrible negative poor hate disgusting waste dislike scam downgrade worse worst"

# Vectorize the reference texts
positive_reference_vector = vectorizer.transform([positive_reference_text])
negative_reference_vector = vectorizer.transform([negative_reference_text])


@app.route('/predict', methods=['POST'])
def predict():
    review = request.json['review']
    cleaned_review = clean_text(review)
    vectorized_review = vectorizer.transform([cleaned_review])
    # prediction = model.predict(vectorized_review)[0]
    # return jsonify({'sentiment': prediction})

  # Calculate cosine similarity
    positive_similarity, negative_similarity = calculate_cosine_similarity(
        vectorized_review, positive_reference_vector, negative_reference_vector)

    # Adjust sentiment prediction based on cosine similarity
    if positive_similarity > negative_similarity:
        prediction = 1  # Positive
    else:
        prediction = 0  # Negative

    return jsonify({'sentiment': 'positive' if prediction == 1 else 'negative'})


@app.route('/recommend', methods=['GET'])
def recommend():
    processed_reviews_path = os.path.join(current_dir, 'processed_reviews.csv')
    reviews = pd.read_csv(processed_reviews_path)

    recommended_products = reviews[reviews['sentiment']
                                   == 1]['product_id'].unique()
    return jsonify(recommended_products.tolist())


@app.route('/cosine_similarity', methods=['POST'])
def cosine_similarity_endpoint():
    review = request.json['review']
    cleaned_review = clean_text(review)
    vectorized_review = vectorizer.transform([cleaned_review])

    positive_similarity, negative_similarity = calculate_cosine_similarity(
        vectorized_review, positive_reference_vector, negative_reference_vector)

    return jsonify({
        'positive_similarity': positive_similarity,
        'negative_similarity': negative_similarity
    })


if __name__ == '__main__':
    app.run(debug=True)
