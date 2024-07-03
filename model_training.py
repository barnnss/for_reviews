import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


def train_model(data_path):
    reviews = pd.read_csv(data_path)
    reviews = reviews[reviews['sentiment'] != 'neutral']

    # Handle missing values
    # Remove rows with NaN in 'cleaned_review'
    reviews = reviews.dropna(subset=['cleaned_review'])

    X = reviews['cleaned_review']
    y = reviews['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print(classification_report(y_test, y_pred))
    return model, vectorizer


if __name__ == '__main__':
    # data_path = 'processed_reviews.csv'
    model, vectorizer = train_model('processed_reviews.csv')
    import pickle
    with open('sentiment_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    with open('vectorizer.pkl', 'wb') as vec_file:
        pickle.dump(vectorizer, vec_file)
