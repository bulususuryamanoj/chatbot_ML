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

# ===========================================================
# LOAD PYTHON RESPONSES
# ===========================================================
with open('data/python.json','r',encoding='utf-8') as f:
    python_data = json.load(f)
python_responses = {}
for intent in python_data:
    python_responses[intent['tag'] ]=  intent['response']

# =========================================================
# LOAD STATISTICS AND MATHEMATICS RESPONSES
# =========================================================
with open('data/stat_mat.json','r',encoding='utf-8') as f:
    stat_mat_data = json.load(f)
stat_mat_response = {}
for intent in stat_mat_data:
    stat_mat_response[intent['tag']] = intent['responses'][0]
# =========================================================
# LOAD DATA MODELLING AND EDA RESPONSE
# =========================================================
with open('data/dm_eda.json','r',encoding='utf-8') as f:
    dm_eda_data = json.load(f)
dm_eda_responses= {}
for intent in dm_eda_data:
    dm_eda_responses[intent['tag']] = intent['responses'][0]

# =========================================================
# LOAD SQL RESPONSES
# =========================================================
with open('data/sql.json','r',encoding='utf-8') as f:
    sql_data = json.load(f)
sql_responses ={}
for intent in sql_data:
    sql_responses[intent['tag']] = intent['responses'][0]

# ========================================================
# LOAD VISUALISATION DATA
# ========================================================
with open('data/viz.json','r',encoding='utf-8') as f:
    viz_data = json.load(f)
viz_responses = {}
for intent in viz_data:
    viz_responses[intent['tag']] = intent['responses'][0]
# ========================================================
# FORMAT STRUCTURED RESPONSE
# ========================================================
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
    # LinearSVC Prediction
    # -------------------------
    input_vector = vectorizer.transform([user_message])

    prediction = model.predict(input_vector)
    predicted_tag = label_encoder.inverse_transform(prediction)[0]

    # Confidence score (from decision function)
    decision_scores = model.decision_function(input_vector)

    if len(decision_scores.shape) > 1:
        confidence = float(np.max(decision_scores))
    else:
        confidence = float(decision_scores[0])

    # -------------------------
    # Collect All Possible Answers
    # -------------------------
    best_concept = find_best_concept(user_message)

    all_answers = []

    if predicted_tag in python_responses:
        all_answers.append(python_responses[predicted_tag])

    if predicted_tag in aiml_responses:
        all_answers.append(aiml_responses[predicted_tag])

    if predicted_tag in stat_mat_response:
        all_answers.append(stat_mat_response[predicted_tag])

    if predicted_tag in dm_eda_responses:
        all_answers.append(dm_eda_responses[predicted_tag])

    if predicted_tag in sql_responses:
        all_answers.append(sql_responses[predicted_tag])

    if predicted_tag in viz_responses:
        all_answers.append(viz_responses[predicted_tag])

    if best_concept and best_concept in concepts:
        all_answers.append(concepts[best_concept])

    if all_answers:
        final_answer = max(all_answers, key=len)
    else:
        final_answer = "I understand the topic but don't have detailed information yet."

    return jsonify({
        "response": final_answer,
        "confidence": round(confidence, 3)
    })
# ==========================================================
# RUN SERVER
# ==========================================================
if __name__ == "__main__":
    app.run(debug=True)