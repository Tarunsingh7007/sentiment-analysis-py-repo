import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load CSV
df = pd.read_csv("product_reviews.csv")  # simple read, no engine or encoding needed now

# Encode sentiments: text → numbers
df['Sentiment'] = df['Sentiment'].map({'Negative': 0, 'Neutral': 1, 'Positive': 2})

# Vectorize the review text
vec = CountVectorizer()
X = vec.fit_transform(df['Review_Text'])
y = df['Sentiment']

# Train model
model = MultinomialNB()
model.fit(X, y)

# Save model and vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vec, "vectorizer.pkl")

print("✅ Model and Vectorizer saved successfully!")
