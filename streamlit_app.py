import streamlit as st
from PIL import Image
from keras.models import load_model
from keras.utils.image_utils import load_img, img_to_array
import numpy as np

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


def model_predict(image, model):
    image = image.resize((224,224))
    x = img_to_array(image)
    x = x / 255
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x)
    pred = np.argmax(pred, axis=1)[0]
    pred = car_dict[pred]
    return ("The Car is " + str(pred))


def main():
    st.title("Car Model Classification")
    result = ""
    
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        result = model_predict(image, model)
        st.success(result)
        st.image(image, caption='Uploaded Image')
    


if __name__ == "__main__":
    main()