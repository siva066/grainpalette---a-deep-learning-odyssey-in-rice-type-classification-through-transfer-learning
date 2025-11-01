from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model("model/rice_model.h5")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files['file']
    img = Image.open(file).resize((224, 224))
    img_array = np.expand_dims(np.array(img)/255.0, axis=0)
    prediction = model.predict(img_array)
    predicted_class = int(np.argmax(prediction, axis=1)[0])
    return jsonify({"class": predicted_class})

# For local testing only
if __name__ == "__main__":
    app.run()
