import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Download stopwords (only first time)
nltk.download('stopwords')

# Display all rows/columns in pandas
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

print("Environment Ready")

# Load CSV
df = pd.read_csv("Clean.Resume.csv")

# If 'Category' column does not exist, create temporary demo categories
if 'Category' not in df.columns:
    # Example: assign ML/Data/Software/other in a round-robin fashion
    categories = ['ML', 'Data', 'Software', 'Cybersecurity']
    df['Category'] = [categories[i % len(categories)] for i in range(len(df))]

print("CSV Loaded:")
print(df)

# Text Cleaning
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()  # lowercase
    text = re.sub(r'[^a-z\s]', ' ', text)  # replace non-letters with space
    words = [w for w in text.split() if w not in stop_words]
    return ' '.join(words)

df['Skills_Clean'] = df['Skills'].apply(clean_text)
print("Text Cleaning Done")
print(df[['Skills', 'Skills_Clean']])

# TF-IDF Vectorization
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['Skills_Clean'])
print("TF-IDF Feature Matrix Created")
print("Shape:", X.shape)

# Encode Labels
le = LabelEncoder()
y = le.fit_transform(df['Category'])
print("Labels Encoded:", y)

# Train/Test Split
# For small datasets, use all data for training/testing to avoid zero-support issues
# When dataset grows, switch to train_test_split for proper evaluation
X_train, X_test, y_train, y_test = X, X, y, y

# Train Model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Predict on Test Set
y_pred = model.predict(X_test)

# Full Classification Report
all_labels = list(le.classes_)
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=range(len(all_labels)),
    target_names=all_labels,
    zero_division=0
))

# Predict New Resume
new_resume = "Python, Machine Learning, SQL"
new_resume_clean = clean_text(new_resume)
new_tfidf = tfidf.transform([new_resume_clean])
predicted_category = le.inverse_transform(model.predict(new_tfidf))[0]
print("\nPredicted Category for new resume:", predicted_category)

df['Category'] = df['Category'].str.strip()

# Make all category names consistent (lowercase or title case)
df['Category'] = df['Category'].str.title()  # e.g., 'ml' -> 'Ml', 'data' -> 'Data'
print("Unique Categories:", df['Category'].unique())
