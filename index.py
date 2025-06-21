from flask import Flask, render_template, request, redirect, flash
from werkzeug.utils import secure_filename
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
app.secret_key = "secret key"

# Model saved with Keras model.save()
MODEL_PATH ='best_model_VGG.h5'

# Load trained model
model = load_model(MODEL_PATH)


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    x=x/255.0
    x = np.expand_dims(x, axis=0)
  
    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    
    if preds == 0:
        preds = "Viral Pneumonia"
    elif preds == 1:
        preds = "Covid"
    else:
        preds = "Normal"
    
    return preds


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def submit_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file:
            # Get the file from post request
            f = request.files['file']

            # Save the file to ./uploads
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(
                basepath, 'static', 'uploads', secure_filename(f.filename))
            f.save(file_path)
            
            image_file = 'uploads/' + str(f.filename)
            
            # Make prediction
            preds = model_predict(file_path, model)
            
            flash(preds)
            return render_template('predict.html', files=image_file)


if __name__ == "__main__":
    app.debug = True
    app.run()