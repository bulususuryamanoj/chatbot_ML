from flask import Flask, request, jsonify, render_template
import pickle
import json
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# ------------------------------------------------
# LOAD ML MODELS
# ------------------------------------------------
model = pickle.load(open(r"model/chatbot_model.pkl",'rb'))
vectorizer = pickle.load(open(r"model/vectorizer.pkl",'rb'))
label_encoder = pickle.load(open(r"model/label_encoder.pkl",'rb'))

# ------------------------------------------------
# LOAD CONCEPT DATA
# ------------------------------------------------
with open("data/content.json") as file:
    concepts = json.load(file)

concept_names = list(concepts.keys())

# ⭐ Improve semantic matching
concept_vectors = vectorizer.transform([c.replace("_"," ") for c in concept_names])

# ------------------------------------------------
# FORMAT RESPONSE (MOVE OUTSIDE ROUTE)
# ------------------------------------------------
def format_response(concept_name, concept_data):

    text = f"📘 Concept: {concept_name.title()}\n\n"

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

    return text

# ------------------------------------------------
# SEMANTIC SEARCH FUNCTION
# ------------------------------------------------
def find_best_concept(user_text):

    user_vector = vectorizer.transform([user_text])
    similarity_scores = cosine_similarity(user_vector,concept_vectors)

    best_index = similarity_scores.argmax()
    best_score = similarity_scores[0][best_index]

    if best_score < 0.15:
        return None

    return concept_names[best_index]

# ------------------------------------------------
# ROUTES
# ------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route('/chat',methods=['POST'])
def chat():

    data = request.get_json()
    user_message = data.get("message")

    # ML topic prediction (optional use)
    input_vector = vectorizer.transform([user_message])
    prediction = model.predict(input_vector)

    # Semantic concept search
    best_concept = find_best_concept(user_message)

    if best_concept:
        answer = format_response(best_concept, concepts[best_concept])
    else:
        answer = "Sorry, The LLM Is under training for some concepts"

    return jsonify({
        "response": answer
    })

# ------------------------------------------------
# RUN APP
# ------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)