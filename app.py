from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        pixels = request.form["pixels"]
        pixels = list(map(int, pixels.split(",")))

        if len(pixels) != 784:
            return "Error: Please enter exactly 784 pixel values."

        final_input = np.array(pixels).reshape(1, -1)
        prediction = model.predict(final_input)

        return f"Predicted Digit: {prediction[0]}"

    except:
        return "Invalid Input"

if __name__ == "__main__":
    app.run(debug=True)
