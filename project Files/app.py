from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

MODEL_PATH = 'rice_model.h5'
model = load_model(MODEL_PATH)

class_names = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

UPLOAD_FOLDER = 'static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        img = image.load_img(filepath, target_size=(224, 224))
        x = image.img_to_array(img) / 255.0
        x = np.expand_dims(x, axis=0)

        pred = model.predict(x)[0]
        idx = np.argmax(pred)
        result = f"{class_names[idx]} ({round(pred[idx]*100, 2)}%)"

        return render_template('result.html', prediction=result, img_path=filepath)

if __name__ == '__main__':
    app.run(debug=True)