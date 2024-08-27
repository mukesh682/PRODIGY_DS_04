import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Load dataset
df = pd.read_csv('twitter_training.csv')

# Data Preprocessing
df.dropna(inplace=True)
df['cleaned_tweet'] = df['tweet'].str.replace(r'@\w+', '')  # Remove mentions
df['cleaned_tweet'] = df['cleaned_tweet'].str.replace(r'#\w+', '')  # Remove hashtags
df['cleaned_tweet'] = df['cleaned_tweet'].str.replace(r'http\S+|www.\S+', '')  # Remove URLs
df['cleaned_tweet'] = df['cleaned_tweet'].str.replace(r'\W', ' ')  # Remove special characters
df['cleaned_tweet'] = df['cleaned_tweet'].str.lower()

# Feature extraction
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['cleaned_tweet']).toarray()
y = df['sentiment']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training with Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions and Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.show()

# Feature Importance
importance = model.coef_[0]
plt.figure(figsize=(100, 100))
plt.barh(np.arange(len(importance)), importance, align='center')
plt.yticks(np.arange(len(importance)), tfidf.get_feature_names_out())
plt.xlabel('Feature Importance')
plt.title('Feature Importance in Logistic Regression')
plt.show()
