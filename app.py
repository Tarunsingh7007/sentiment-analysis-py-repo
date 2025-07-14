from flask import Flask, render_template, request
import joblib

# Flask app create
app = Flask(__name__)

# Load trained model and vectorizer
model = joblib.load("model.pkl")
vec = joblib.load("vectorizer.pkl")

# Label mapping
label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    if request.method == "POST":
        review = request.form["review"]
        vect_input = vec.transform([review])
        pred = model.predict(vect_input)[0]
        result = label_map[pred]
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
