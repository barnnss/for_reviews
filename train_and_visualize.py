import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle


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


def preprocess_data(file_path):
    reviews = pd.read_csv(file_path)
    reviews = reviews.dropna(subset=['reviewText'])
    reviews['cleaned_review'] = reviews['reviewText'].apply(
        lambda x: clean_text(str(x)))
    reviews['sentiment'] = reviews['overall'].apply(
        lambda x: 'positive' if x > 3 else ('negative' if x < 3 else 'neutral'))
    return reviews


def visualize_data_distribution(reviews):
    sns.countplot(x='sentiment', data=reviews)
    plt.title('Sentiment Distribution')
    plt.show()


def generate_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400,
                          background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred, labels=['negative', 'positive'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[
                'negative', 'positive'], yticklabels=['negative', 'positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


def train_model(data_path):
    reviews = preprocess_data(data_path)
    reviews = reviews[reviews['sentiment'] != 'neutral']

    # Visualize data distribution
    visualize_data_distribution(reviews)

    # Generate word cloud
    text = ' '.join(reviews['cleaned_review'].dropna())
    generate_word_cloud(text)

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

    # Evaluate model performance
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred)

    # Save the model and vectorizer
    with open('sentiment_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    with open('vectorizer.pkl', 'wb') as vec_file:
        pickle.dump(vectorizer, vec_file)


if __name__ == '__main__':
    data_path = 'processed_reviews.csv'
    train_model(data_path)
