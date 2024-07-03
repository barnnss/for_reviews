from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')


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
    reviews['cleaned_review'] = reviews['reviewText'].apply(
        lambda x: clean_text(str(x)))
    reviews['sentiment'] = reviews['overall'].apply(
        lambda x: 'positive' if x > 3 else ('negative' if x < 3 else 'neutral'))
    return reviews


if __name__ == '__main__':
    file_path = 'amazon.csv'
    processed_reviews = preprocess_data('processed_reviews.csv')
    processed_reviews.to_csv('processed_reviews.csv', index=False)
