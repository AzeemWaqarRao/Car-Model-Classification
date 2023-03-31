from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
import os
import sys
import glob
import re
import numpy as np
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras.utils.image_utils import load_img, img_to_array



app = Flask(__name__)
model_path = "resnet.h5"
model = load_model(model_path)
car_dict = {0: 'Alfa Romeo Stelvio',
 1: 'Aston Martin DB11',
 2: 'Aston Martin DBS',
 3: 'Aston Martin Valkyrie',
 4: 'Aston Martin Vantage',
 5: 'Aston Martin Vulcan',
 6: 'Audi A3',
 7: 'Audi A6',
 8: 'Audi E-tron GT',
 9: 'Audi R8',
 10: 'BMW 3-series',
 11: 'BMW 7-series',
 12: 'BMW x7',
 13: 'Bentley Bentayga',
 14: 'Bentley Continental',
 15: 'Bugatti Centidieci',
 16: 'Bugatti Chiron',
 17: 'Bugatti Divo',
 18: 'Bugatti La Voiture Noire',
 19: 'Buggati Veyron',
 20: 'Cadillac Escalade',
 21: 'Corvette ZR',
 22: 'Ferrari 458',
 23: 'Ferrari FF',
 24: 'Ferrari Pininfarina',
 25: 'Jaguar F-type',
 26: 'Jaguar XJ',
 27: 'Koenigsegg CC8S',
 28: 'Koenigsegg CCX',
 29: 'La Ferrari',
 30: 'Lamborghini Gallardo',
 31: 'Lamborghini Murceilago',
 32: 'Lamborghini Veneno',
 33: 'Mustang GT',
 34: 'Pagani Zonda',
 35: 'Porsche 911',
 36: 'Porsche Cayenne',
 37: 'Range Rover Discovery',
 38: 'Renault Duster',
 39: 'Rolls Royce Ghost',
 40: 'Rolls Royce Phantom',
 41: 'Tata Tiago',
 42: 'Toyota Fortuner',
 43: 'Volkswagen Polo',
 44: 'Volkswagen Vento'}


def clearuploads():
    basepath = os.path.dirname(__file__)
    directory = 'uploads'

    # Get all files in the directory
    file_list = os.listdir(os.path.join(basepath,directory))

    # Loop through each file and delete it
    for filename in file_list:
        
        file_path = os.path.join(directory, filename)
        os.remove(file_path)

        
def model_predict(img_path, model):
    img = load_img(img_path, target_size=(224, 224))

    x = img_to_array(img)
    x = x / 255
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x)
    pred = np.argmax(pred, axis=1)[0]
    pred = car_dict[pred]
    clearuploads()
    return ("The Car is " + str(pred))


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('temp.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        # return result
        return render_template("temp.html", prediction_text= str(result))
    return None


if __name__ == '__main__':
    app.run(debug=True)