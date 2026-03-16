import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load cleaned datasets
train = pd.read_csv("data/cleaned_train.csv")
test = pd.read_csv("data/cleaned_test.csv")

# Separate text and labels
X_train = train["statement"]
y_train = train["label"]

X_test = test["statement"]
y_test = test["label"]

# Convert text to numerical vectors
vectorizer = TfidfVectorizer(max_features=5000)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Predict
predictions = model.predict(X_test_vec)

# Evaluate
print(classification_report(y_test, predictions))