import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# load data
train_data = pd.read_csv("train_movie_reviews_data.csv")
test_data = pd.read_csv("test_movie_reviews_data.csv")

# fit a tfidf vectorizor over the training text
tfidf = TfidfVectorizer(ngram_range= (1, 1), max_features=5000, min_df = 100)
tfidf.fit(train_data.text.values)
train_tfidf_matrix = tfidf.transform(train_data.text.values)

# Use a classifier (logistic regression in this case) to feed in data
logreg = LogisticRegression()
logreg.fit(train_tfidf_matrix, train_labels)

# Training score/accuracy
logreg.score(train_tfidf_matrix, train_labels)

# testing score/accuracy
test_tfidf_matrix = tfidf.transform(test_data.text.values)
logreg.score(test_tfidf_matrix, test_labels)

