from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

df = pd.read_csv("dataset.csv")

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["input"])
y = df["response"]

model = MultinomialNB()
model.fit(X, y)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"response": "⚠️ Pesan kosong."})

    
    X_user = vectorizer.transform([user_message])
    prediction = model.predict(X_user)[0]

    return jsonify({"response": prediction})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
