from flask import Flask, request, render_template
import numpy as np
import os 
from keras.models import load_model
import keras.utils as image
import tensorflow as tf
import sys
from PIL import Image
sys.modules['Image'] = Image 
global graph
graph=tf.compat.v1.get_default_graph()
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
model = load_model('model/model_best.h5')
target_img = os.path.join(os.getcwd() , 'static/images')
@app.route('/')
def index_view():
    return render_template('index.html')
#Allow files with extension png, jpg and jpeg
ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png', 'JPG', 'JPEG', 'PNG'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT
           
# Function to load and prepare the image in right shape
def read_image(filename):
    img = image.load_img(filename, target_size=(180,180))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    #x = preprocess_input(x)
    return x
@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename): #Checking file format
            filename = file.filename
            file_path = os.path.join('static/images', filename)
            file.save(file_path)
            img = read_image(file_path) #prepressing method
            class_prediction=model.predict(img) 
            class_pred_per=round(np.max(class_prediction)*100,2)
            if class_pred_per > 100: class_pred_per = 100
            elif class_pred_per < 0: class_pred_per = 0
            classes_x=np.argmax(class_prediction,axis=1)
            if classes_x == 0:
              prediction = "Benign"
            elif classes_x == 1:
              prediction = "Malignant"

            return render_template('predict.html', prediction = prediction,prob=class_pred_per, user_image = file_path)
        else:
            return "Unable to read the file. Please check file extension"
if __name__ == '__main__':
    app.run(debug=True,use_reloader=False, port=8080)