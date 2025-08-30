from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
import os

app = Flask(__name__)

# ====== Load dataset ======
df = pd.read_csv("dataset.csv")

# Pastikan kolomnya bernama 'pertanyaan' dan 'jawaban'
X = df['pertanyaan']
y = df['jawaban']

# ====== Training model ======
vectorizer = TfidfVectorizer()
X_vectors = vectorizer.fit_transform(X)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_vectors, y)

# ====== API endpoint ======
@app.route("/")
def home():
    return "✅ Chatbot API aktif di Railway!"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message")

    if not user_input:
        return jsonify({"error": "Pesan kosong"}), 400

    # Prediksi jawaban
    input_vec = vectorizer.transform([user_input])
    response = model.predict(input_vec)[0]

    return jsonify({"reply": response})

# ====== Run server ======
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
