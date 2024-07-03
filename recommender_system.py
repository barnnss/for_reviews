import pandas as pd
import pickle
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity


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


def recommend_products(data_path, model_path, vectorizer_path):
    reviews = pd.read_csv(data_path)
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    with open(vectorizer_path, 'rb') as vec_file:
        vectorizer = pickle.load(vec_file)

    # Define reference texts for positive and negative sentiments
    # positive_reference_text = "good excellent great amazing positive wonderful"
    # negative_reference_text = "bad terrible awful horrible negative poor"

    # Vectorize the reference texts
    # positive_reference_vector = vectorizer.transform([positive_reference_text])
    # negative_reference_vector = vectorizer.transform([negative_reference_text])

    reviews['cleaned_review'] = reviews['reviewText'].apply(
        lambda x: clean_text(str(x)))
    reviews['vectorized_review'] = vectorizer.transform(
        reviews['cleaned_review'])
    reviews['sentiment'] = model.predict(reviews['vectorized_review'])

    # Vectorize the cleaned reviews
    # vectorized_reviews = vectorizer.transform(reviews['cleaned_review'])

    # Predict sentiments and calculate cosine similarity

    recommended_products = reviews[reviews['sentiment']
                                   == 1]['product_id'].unique()
    return recommended_products


if __name__ == '__main__':
    data_path = 'processed_reviews.csv'
    model_path = 'sentiment_model.pkl'
    vectorizer_path = 'vectorizer.pkl'
    recommended_products = recommend_products(
        data_path, model_path, vectorizer_path)
    print(recommended_products)
