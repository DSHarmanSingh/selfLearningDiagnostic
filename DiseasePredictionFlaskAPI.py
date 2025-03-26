import numpy as np
import joblib
import tensorflow.lite as tflite
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient

# ✅ Load Label Encoder & TF-IDF Vectorizer
label_encoder = joblib.load("LabelEncoder.pkl")
tfidf = joblib.load("tfidf.pkl")

# ✅ Load TF-Lite Model
interpreter = tflite.Interpreter(model_path="DiseasePrediction_DeepLearning.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ✅ Connect to MongoDB Atlas
client = MongoClient("mongodb+srv://DSHarman:<harman.mongodb.ds>@harmandiseaseprediction.wbo3b.mongodb.net/?retryWrites=true&w=majority&appName=harmandiseaseprediction)
db = client["sldds"]
collection = db["user_queries"]

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Welcome to AI-Powered Healthcare API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    symptoms_text = data.get("symptoms", "").strip()

    if not symptoms_text:
        return jsonify({"error": "No symptoms provided!"}), 400

    # ✅ Convert symptoms to TF-IDF vector
    symptoms_vector = tfidf.transform([symptoms_text]).toarray().astype(np.float32)

    # ✅ Run TF-Lite model
    interpreter.set_tensor(input_details[0]['index'], symptoms_vector)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    # ✅ Get the predicted disease
    predicted_label = np.argmax(prediction)
    predicted_disease = label_encoder.inverse_transform([predicted_label])[0]

    # ✅ Save Query in MongoDB
    collection.insert_one({"symptoms": symptoms_text, "predicted_disease": predicted_disease, "user_feedback": None})

    return jsonify({"predicted_disease": predicted_disease})


@app.route('/update', methods=['POST'])
def update():
    data = request.json
    symptoms = data.get("symptoms", "").strip()
    correct_disease = data.get("correct_disease", "").strip()

    if not symptoms or not correct_disease:
        return jsonify({"error": "Symptoms or correct disease missing!"}), 400

    # ✅ Update MongoDB with feedback
    collection.update_one({"symptoms": symptoms}, {"$set": {"user_feedback": correct_disease}})

    # ✅ Reinforcement Learning - Fine-tune Model
    update_model(symptoms, correct_disease)

    return jsonify({"message": "Model updated successfully!"})


def update_model(user_input, correct_disease):
    """Reinforcement Learning: Update model based on user feedback."""
    import tensorflow as tf

    # ✅ Load trained model
    model = tf.keras.models.load_model("DiseasePrediction.h5")

    # ✅ Encode correct label
    correct_label = label_encoder.transform([correct_disease])[0]

    # ✅ Convert input into feature vector
    user_vector = tfidf.transform([user_input]).toarray()

    # ✅ Fine-tune model (one-step training with new data)
    model.fit(user_vector, np.array([correct_label]), epochs=1, verbose=0)

    # ✅ Save updated model
    model.save("DiseasePrediction.h5")
    print("Model updated successfully!")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=10000)
