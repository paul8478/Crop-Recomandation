from flask import Flask, request, render_template
import numpy as np
import pickle
import os

# Load the trained model
with open("crop_recommendation.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load label encoder (to decode crop names)
with open("label_encoder.pkl", "rb") as le_file:
    label_encoder = pickle.load(le_file)

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")  # Load HTML form

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        N = float(request.form["N"])
        P = float(request.form["P"])
        K = float(request.form["K"])
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        ph = float(request.form["ph"])
        rainfall = float(request.form["rainfall"])

        # Create feature array
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

        # Predict crop
        prediction = model.predict(input_data)
        predicted_crop = label_encoder.inverse_transform(prediction)[0]

        # Define image path
        image_filename = f"static/crop-img/{predicted_crop.lower()}.png"  # Assuming all images are lowercase
        # Check if the image exists
        if not os.path.exists(image_filename):
            image_filename = "crop-img/default.png"  # Fallback image if not found

        return render_template(
            "index.html",
            prediction_text=f"Recommended Crop: {predicted_crop}",
            crop_image=image_filename,
        )

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
