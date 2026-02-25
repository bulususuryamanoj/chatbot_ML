from sklearn.metrics.pairwise import cosine_similarity
from flask import render_template,Flask,request,jsonify
import pickle
import json
app = Flask(__name__)
model = pickle.load(open(r"model\chatbot_model.pkl",'rb'))
vectorizer = pickle.load(open(r"model\vectorizer.pkl",'rb'))
label_encoder = pickle.load(open(r"model\label_encoder.pkl"))
with open("data/concepts.json") as file:
    concepts = json.load(file)
concept_names = list(concepts.keys())

# Create semantic vectors for all concepts
concept_vectors = vectorizer.transform(concept_names)
def find_best_concept(user_text):

    user_vector = vectorizer.transform([user_text])

    similarity_scores = cosine_similarity(user_vector, concept_vectors)

    best_index = similarity_scores.argmax()
    best_score = similarity_scores[0][best_index]

    # Confidence threshold
    if best_score < 0.15:
        return None

    return concept_names[best_index]
@app.route("/")
def home():
    return render_template("index.html")
@app.route('/chat',methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get("message")

    # -----------------------------
    # ML Topic Prediction
    # -----------------------------
    input_vector = vectorizer.transform([user_message])
    prediction = model.predict(input_vector)
    topic = label_encoder.inverse_transform(prediction)[0]

    # -----------------------------
    # Semantic Concept Search
    # -----------------------------
    best_concept = find_best_concept(user_message)

    if best_concept:
        answer = concepts[best_concept]
    else:
        answer = "I understand the topic but don't have that concept explanation yet."

    return jsonify({
        "topic": topic,
        "response": answer
    })