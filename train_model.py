import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load real dataset
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

fake['label'] = 'FAKE'
true['label'] = 'REAL'

data = pd.concat([fake, true])
data = data.sample(frac=1).reset_index(drop=True)

X = data['text']
y = data['label']

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vectorized = vectorizer.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_vectorized, y)

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model trained successfully!")
