from flask import Flask, request, jsonify, render_template
import pickle
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# ==========================================================
# LOAD ML MODELS
# ==========================================================
model = pickle.load(open("model/chatbot_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))
label_encoder = pickle.load(open("model/label_encoder.pkl", "rb"))

# ==========================================================
# LOAD CONTENT DATA (Detailed Knowledge Base)
# ==========================================================
with open("data/content.json", "r", encoding="utf-8") as f:
    concepts = json.load(f)

concept_names = list(concepts.keys())

# ==========================================================
# BUILD SEMANTIC VECTORS (STRONG MATCHING)
# ==========================================================
concept_texts = []

for name in concept_names:
    concept_data = concepts[name]
    combined_text = name.replace("_", " ")

    if isinstance(concept_data, dict):
        for value in concept_data.values():
            if isinstance(value, list):
                combined_text += " " + " ".join(value)
            else:
                combined_text += " " + str(value)
    else:
        combined_text += " " + str(concept_data)

    concept_texts.append(combined_text)

concept_vectors = vectorizer.transform(concept_texts)

# ==========================================================
# LOAD AIML RESPONSES
# ==========================================================
with open("data/aiml.json", "r", encoding="utf-8") as f:
    aiml_data = json.load(f)

aiml_responses = {}
for intent in aiml_data:
    aiml_responses[intent["tag"]] = intent["responses"][0]

# ==========================================================
# FORMAT STRUCTURED RESPONSE
# ==========================================================
def format_response(concept_name, concept_data):

    text = f"📘 Concept: {concept_name.replace('_',' ').title()}\n\n"

    if isinstance(concept_data, dict):

        if "level" in concept_data:
            text += f"🎯 Level: {concept_data['level'].title()}\n\n"

        if "definition" in concept_data:
            text += f"📖 Definition:\n{concept_data['definition']}\n\n"

        if "explanation" in concept_data:
            text += f"🧠 Explanation:\n{concept_data['explanation']}\n\n"

        if "example" in concept_data:
            text += f"💡 Example:\n{concept_data['example']}\n\n"

        if "key_points" in concept_data:
            points = ", ".join(concept_data["key_points"])
            text += f"🔑 Key Points:\n{points}\n\n"

        if "interview_tip" in concept_data:
            text += f"🎤 Interview Tip:\n{concept_data['interview_tip']}"

    else:
        text += str(concept_data)

    return text


# ==========================================================
# SEMANTIC SEARCH
# ==========================================================
def find_best_concept(user_text):

    user_vector = vectorizer.transform([user_text])
    similarity_scores = cosine_similarity(user_vector, concept_vectors)

    best_index = similarity_scores.argmax()
    best_score = similarity_scores[0][best_index]

    if best_score < 0.20:   # slightly higher threshold for better quality
        return None

    return concept_names[best_index]


# ==========================================================
# ROUTES
# ==========================================================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json()
    user_message = data.get("message")

    # -------------------------
    # Intent Prediction
    # -------------------------
    input_vector = vectorizer.transform([user_message])
    probs = model.predict_proba(input_vector)[0]

    max_prob = probs.max()
    predicted_tag = label_encoder.inverse_transform(
        [probs.argmax()]
    )[0]

    # -------------------------
    # PRIORITY 1 → CONTENT.JSON
    # -------------------------
    best_concept = find_best_concept(user_message)

    if best_concept and predicted_tag not in aiml_responses:
        final_answer = format_response(
            best_concept,
            concepts[best_concept]
        )

    # -------------------------
    # PRIORITY 2 → AIML FALLBACK
    # -------------------------
    elif predicted_tag in aiml_responses:
        final_answer = aiml_responses[predicted_tag]

    # -------------------------
    # FALLBACK
    # -------------------------
    else:
        final_answer = "⚠️ The assistant is still learning this concept."

    return jsonify({
        "response": final_answer,
        "confidence": round(float(max_prob), 3)
    })


# ==========================================================
# RUN SERVER
# ==========================================================
if __name__ == "__main__":
    app.run(debug=True)