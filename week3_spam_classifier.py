import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# Create SMS dataset
data = {
    "message": [
        "Congratulations you have won a free prize",
        "Claim your free cash reward now",
        "You have been selected to win a lottery",
        "Urgent call now to claim your prize",
        "Free entry in a contest win cash",
        "You won a free mobile phone",
        "Claim your reward before it expires",
        "Congratulations claim your free gift",
        "Win money now click here",
        "Exclusive offer claim your prize now",

        "Hi how are you doing today",
        "Can we meet tomorrow",
        "Please send me the assignment",
        "Are you coming to college today",
        "Lets have lunch together",
        "Call me when you reach home",
        "Dont forget the meeting tomorrow",
        "What time is the class",
        "I will see you in the evening",
        "Can you help me with this project"
    ],

    "label": [
        "spam", "spam", "spam", "spam", "spam",
        "spam", "spam", "spam", "spam", "spam",

        "ham", "ham", "ham", "ham", "ham",
        "ham", "ham", "ham", "ham", "ham"
    ]
}


df = pd.DataFrame(data)

print("===== DATASET =====")
print(df)


# Select feature and target
X = df["message"]
y = df["label"]


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Text preprocessing using TF-IDF
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# Create Naive Bayes model
model = MultinomialNB()


# Train the model
model.fit(X_train_tfidf, y_train)


# Make predictions
y_pred = model.predict(X_test_tfidf)


print("\n===== ACTUAL LABELS =====")
print(list(y_test))

print("\n===== PREDICTED LABELS =====")
print(list(y_pred))


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print("\n===== MODEL EVALUATION =====")
print("Accuracy:", accuracy)

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, y_pred))


# Test the model with new messages
new_messages = [
    "Congratulations you won free cash",
    "Are we meeting after class today",
    "Claim your free reward now",
    "Please send me the notes"
]

new_messages_tfidf = vectorizer.transform(new_messages)
predictions = model.predict(new_messages_tfidf)

print("\n===== NEW MESSAGE PREDICTIONS =====")

for message, prediction in zip(new_messages, predictions):
    print(f"Message: {message}")
    print(f"Prediction: {prediction}\n")