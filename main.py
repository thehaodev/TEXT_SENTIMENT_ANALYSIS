import numpy as np
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import contractions
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

nltk.download('stopwords')
nltk.download('wordnet')


def expand_contractions(text):
    return contractions.fix(text)


def preprocess_text(text):
    # Removing html tags
    wl = WordNetLemmatizer()
    soup = BeautifulSoup(text, "html.parser")

    # Expanding chatwords and contracts clearing contractions
    text = soup.get_text()
    text = expand_contractions(text)

    emoji_clean = re.compile("["
                             u"\U0001F600-\U0001F64F"
                             u"\U0001F300-\U0001F5FF"
                             u"\U0001F680-\U0001F6FF"
                             u"\U0001F1E0-\U0001F1FF"
                             u"\U00002702-\U000027B0"
                             u"\U000024C2-\U0001F251"
                             "]+", flags=re.UNICODE)

    text = emoji_clean.sub(r'', text)
    text = re.sub(r'\.(?=\S)', '. ', text)
    text = re.sub(r'http\S+', '', text)

    # remove punctuation and make text lowercase
    text = "".join([
        word.lower() for word in text if word not in string.punctuation
    ])
    # lemmatize
    stop = set(stopwords.words('english'))
    text = " ".join([wl.lemmatize(word) for word in text.split() if word not in stop and word.isalpha()])

    return text


def func(pct, all_values):
    absolute = int(pct / 100. * np.sum(all_values))
    return "{:.1f}%\n({:d})".format(pct, absolute)


def data_analysis(data_frame):
    freq_pos = len(data_frame[data_frame['sentiment'] == 'positive'])
    freq_neg = len(data_frame[data_frame['sentiment'] == 'negative'])

    data = [freq_pos, freq_neg]

    pie, _ = plt.subplots(figsize=[11, 7])
    plt.pie(x=data, autopct=lambda pct: func(pct, data), explode=[0.0025] * 2,
            pctdistance=0.5, colors=[sns.color_palette()[0], 'tab:red'], textprops={'fontsize': 16})

    labels = [r'Positive', r'Negative']
    plt.legend(labels, loc="best", prop={'size': 14})
    pie.savefig("PieChart.png")
    plt.show()


def length_sample_fig(data_frame):
    columns_word_length = "words length"

    words_len = data_frame['review'].str.split().map(lambda x: len(x))
    data_frame_temp = data_frame.copy()
    data_frame_temp[columns_word_length] = words_len

    # hist_positive
    sns.displot(
        data=data_frame_temp[data_frame_temp['sentiment'] == 'positive'],
        x=columns_word_length,
        hue='sentiment',
        kde=True, height=7, aspect=1.1, legend=False
    ).set(title='Words in positive reviews')

    # hist_negative
    sns.displot(
        data=data_frame_temp[data_frame_temp['sentiment'] == 'negative'],
        x=columns_word_length,
        hue='sentiment',
        kde=True, height=7, aspect=1.1, legend=False, palette=['red']
    ).set(title='Words in positive reviews')

    plt.figure(figsize=(7, 7.1))
    sns.kdeplot(
        data=data_frame_temp,
        x=columns_word_length,
        hue='sentiment',
        fill=True,
        palette=[sns.color_palette()[0], 'red']
    ).set(title='Words in reviews')
    plt.legend(title='Sentiment', labels=['negative', 'positive'])

    plt.show()


def prediction_sklearn(data_frame):
    label_encode = LabelEncoder()
    y_data = label_encode.fit_transform(data_frame['sentiment'])
    x_data = data_frame['review']

    # Divide to train and test data
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

    # Transform to vector
    tfidf_vectorizer = TfidfVectorizer(max_features=10000)
    tfidf_vectorizer.fit(x_train, y_train)

    x_train_encoded = tfidf_vectorizer.transform(x_train)
    x_test_encoded = tfidf_vectorizer.transform(x_test)

    # Train and Evaluate model with Decision Tree
    dt_classifier = DecisionTreeClassifier(
        criterion='entropy',
        random_state=42,
        ccp_alpha=0.0
    )
    dt_classifier.fit(x_train_encoded, y_train)
    y_pred = dt_classifier.predict(x_test_encoded)
    accuracy_score(y_pred, y_test)

    # Train and Evaluate model with RandomForest
    rf_classifier = RandomForestClassifier(
        random_state=42,
        min_samples_leaf=1,
        max_features='sqrt'
    )
    rf_classifier.fit(x_train_encoded, y_train)
    y_pred = rf_classifier.predict(x_test_encoded)
    accuracy_score(y_pred, y_test)


def run():
    df = pd.read_csv('IMDB-Dataset.csv')

    # Remove duplicate rows
    df = df.drop_duplicates()
    df['review'] = df['review'].apply(preprocess_text)

    data_analysis(data_frame=df)
    length_sample_fig(data_frame=df)
    prediction_sklearn(data_frame=df)


run()
