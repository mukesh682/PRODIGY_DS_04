import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Load training dataset
df = pd.read_csv('twitter_training.csv')

# Load validation dataset
validation_df = pd.read_csv('validation_twitter.csv')

# Check the column names to verify the text column name
print(df.columns)
print(validation_df.columns)

# Data Preprocessing for Training Data
df.dropna(inplace=True)
df['cleaned_tweet'] = df['tweet'].str.replace(r'@\w+', '', regex=True)  # Remove mentions
df['cleaned_tweet'] = df['cleaned_tweet'].str.replace(r'#\w+', '', regex=True)  # Remove hashtags
df['cleaned_tweet'] = df['cleaned_tweet'].str.replace(r'http\S+|www.\S+', '', regex=True)  # Remove URLs
df['cleaned_tweet'] = df['cleaned_tweet'].str.replace(r'\W', ' ', regex=True)  # Remove special characters
df['cleaned_tweet'] = df['cleaned_tweet'].str.lower()

# Data Preprocessing for Validation Data
validation_df.dropna(inplace=True)
validation_df['cleaned_tweet'] = validation_df['tweet'].str.replace(r'@\w+', '', regex=True)  # Remove mentions
validation_df['cleaned_tweet'] = validation_df['cleaned_tweet'].str.replace(r'#\w+', '', regex=True)  # Remove hashtags
validation_df['cleaned_tweet'] = validation_df['cleaned_tweet'].str.replace(r'http\S+|www.\S+', '', regex=True)  # Remove URLs
validation_df['cleaned_tweet'] = validation_df['cleaned_tweet'].str.replace(r'\W', ' ', regex=True)  # Remove special characters
validation_df['cleaned_tweet'] = validation_df['cleaned_tweet'].str.lower()

# Feature extraction
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['cleaned_tweet']).toarray()
y = df['sentiment']

# Prepare validation data for testing
X_validation = tfidf.transform(validation_df['cleaned_tweet']).toarray()
y_validation = validation_df['sentiment']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training with Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions and Evaluation on Test Data
y_pred = model.predict(X_test)
print("Classification Report on Test Data:\n", classification_report(y_test, y_pred))

# Confusion Matrix for Test Data
cm_test = confusion_matrix(y_test, y_pred)
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix on Test Data')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Predictions and Evaluation on Validation Data
y_val_pred = model.predict(X_validation)
print("Classification Report on Validation Data:\n", classification_report(y_validation, y_val_pred))

# Confusion Matrix for Validation Data
cm_validation = confusion_matrix(y_validation, y_val_pred)
sns.heatmap(cm_validation, annot=True, fmt='d', cmap='Greens')
plt.title('Confusion Matrix on Validation Data')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Feature Importance - Top N Features
N = 20  # Number of top features to display
importance = model.coef_[0]  # Coefficients of the features

# Get the indices of the top N features
indices = np.argsort(importance)[-N:]
top_features = [tfidf.get_feature_names_out()[i] for i in indices]
top_importances = importance[indices]

# Plot the top N important features
plt.figure(figsize=(10, 6))
plt.barh(range(len(top_importances)), top_importances, align='center')
plt.yticks(range(len(top_importances)), top_features)
plt.xlabel('Feature Importance')
plt.title(f'Top {N} Important Features in Logistic Regression')
plt.show()
