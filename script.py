import numpy as np
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import nltk
nltk.download('stopwords')

# Load news dataset containing text data and category labels
news = load_files('data', encoding='utf-8', decode_error='replace')

# Split the data into features (text) and target (category)
x = np.asarray(news.data)  # Convert text data to numpy array
y = np.asarray(news.target)  # Convert category labels to numpy array

# Split the data into training and test sets with 30% of the data used for testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=75)

# Initialize the TfidfVectorizer with English stop words and ignore decode errors
vectorizer = TfidfVectorizer(stop_words='english', decode_error='ignore', max_features=1000)

# Fit and transform the text data with TfidfVectorizer
x_train_vectors = vectorizer.fit_transform(x_train)  # Transform training data
x_test_vectors = vectorizer.transform(x_test)  # Transform test data

# Initialize models for Random Forest and Naive Bayes
model1 = RandomForestClassifier(n_estimators=1000, random_state=1, max_depth=100)
model2 = MultinomialNB()

# Train models
model1.fit(x_train_vectors, y_train)
model2.fit(x_train_vectors, y_train)

# Calculate accuracy scores of each model
accuracy_score_rf = model1.score(x_test_vectors, y_test)
accuracy_score_nb = model2.score(x_test_vectors, y_test)

print("\nRandom Forest Model Accuracy Score: ", accuracy_score_rf)
print("\nNaive Bayes Model Accuracy Score: ", accuracy_score_nb)
