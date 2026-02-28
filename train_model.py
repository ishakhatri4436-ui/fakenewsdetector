import pandas as pd
import numpy as np
import re
import string
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ===============================
# 1. Load Dataset
# ===============================

# Make sure your CSV has columns: text,label
df = pd.read_csv("data/news.csv")

# Drop missing values
df = df.dropna()

print("Dataset Loaded Successfully")
print("Total samples:", len(df))


# ===============================
# 2. Clean Text Function
# ===============================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)            # remove URLs
    text = re.sub(r"\[.*?\]", "", text)           # remove brackets
    text = re.sub(r"<.*?>+", "", text)            # remove HTML tags
    text = re.sub("[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub("\n", " ", text)
    text = re.sub("\w*\d\w*", "", text)           # remove words with numbers
    return text


df["text"] = df["text"].apply(clean_text)


# ===============================
# 3. Features & Labels
# ===============================

X = df["text"]
y = df["label"]   # 0 = Fake, 1 = Real


# ===============================
# 4. Train Test Split
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ===============================
# 5. TF-IDF Vectorization
# ===============================

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7,
    max_features=5000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# ===============================
# 6. Model Training
# ===============================

model = LogisticRegression(max_iter=2000)

model.fit(X_train_vec, y_train)


# ===============================
# 7. Evaluation
# ===============================

y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# ===============================
# 8. Save Model & Vectorizer
# ===============================

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\n✅ model.pkl and vectorizer.pkl saved successfully!")
