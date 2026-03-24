import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

# load dataset
df = pd.read_csv("data/student_stress.csv")

# convert stress labels
df["stress"] = df["stress"].map({
    "Low": 0,
    "Medium": 1,
    "High": 2
})

# features and target
X = df[["study_hours", "sleep_hours", "screen_time", "difficulty"]]
y = df["stress"]

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# check accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")

# save model
os.makedirs("model", exist_ok=True)

with open("model/model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model trained successfully")