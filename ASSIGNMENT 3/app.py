from flask import Flask, request, jsonify, render_template_string
import pickle
import warnings
import os
from score import score  # Importing from score.py in the same folder

warnings.filterwarnings("ignore")

# Initialize Flask application
app = Flask(__name__)

# Load the pre-trained model
model = pickle.load(open("model.pkl", "rb"))  # Assuming model.pkl is in the same directory

# HTML template for simple UI
template_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SMS Spam Classifier</title>
</head>
<body>
    <h1>SMS Spam Classifier</h1>
    <form action="/classify" method="post">
        <label for="message">Enter Message:</label>
        <input type="text" id="message" name="message">
        <input type="submit" value="Predict">
    </form>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(template_html)

@app.route("/classify", methods=["POST"])
def classify_message():
    input_text = request.form["message"]
    prediction_label, confidence_score = score(input_text, model, threshold=0.50)

    response_payload = {
        "result": prediction_label,
        "confidence": confidence_score,
        "note": "True = Spam, False = Ham"
    }

    return jsonify(response_payload)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)  
