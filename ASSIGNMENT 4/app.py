from flask import Flask, request, jsonify, render_template_string
from score import score  # Make sure score.py is in same directory
from nltk import download
import pickle
import warnings

warnings.filterwarnings("ignore")


# Download NLTK resources
download('punkt')
download('stopwords')
download('wordnet')

model = pickle.load(open("model.pkl", "rb"))  # Assuming model.pkl is in the same directory

# Sample HTML template (replace with your actual template)
template_html = '''
<!doctype html>
<title>Spam Classifier</title>
<h2>Enter a message:</h2>
<form method=post action="/classify">
  <input type=text name=message>
  <input type=submit value=Classify>
</form>
'''

# Initialize Flask app
spam_classification_app = Flask(__name__)

@spam_classification_app.route("/", methods=["GET"])
def index():
    return render_template_string(template_html)

@spam_classification_app.route("/classify", methods=["POST"])
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
    spam_classification_app.run(host="0.0.0.0", port=5000)
