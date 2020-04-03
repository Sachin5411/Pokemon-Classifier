

from __future__ import division, print_function

import sys
import os
import glob
import re
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import pickle
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from keras import backend as K

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()

poke_index={0:'Pikachu',1:'Bulbasaur',2:'Charmander'}

# Load your trained model

def model_predict(img_path):
	img=image.load_img(img_path,target_size=(120,120,3))
	
	print("IMage loaded ")
	#img=img.reshape(1,120,120,3)
	#print("Reshaped as ",img.shape)
	img_arr=image.img_to_array(img)
	
	print("Image converted to array ")
	img_list=[]
	img_list.append(img_arr)
	img_s=np.array(img_list)
	print("Reshaped as ",img_s.shape)
	#img_s=img_s.reshape(1,120,120,3)
	model2 = pickle.load(open('model2.pkl','rb')) 
	pred=model2.predict(img_s)
	print("predictions are ",pred)
	m=np.argmax(pred)
	print("Argmax  ",m)
	preds=poke_index[m]
	print("returning predicitons ",preds)
	return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        print("path saved as --------  ", file_path)
        # Make prediction
        preds = model_predict(file_path)
        K.clear_session()

        result = str(preds)               # Convert to string
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)



