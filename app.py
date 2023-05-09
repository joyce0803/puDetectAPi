import base64
import fileinput
import os.path

import cv2
from flask import Flask, render_template, request
import io
from patchify import patchify,unpatchify
from PIL import Image
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.utils import normalize, img_to_array, load_img
from keras.models import load_model
from werkzeug.utils import secure_filename

from matplotlib import pyplot as plt

app=Flask(__name__)
patch_size=256
scaler=MinMaxScaler()
pred_model=load_model('building_detection_more_epochs.hdf5')
UPLOAD_FOLDER = 'C:/Users/Joyce Merin/PycharmProjects/BuildingDetection/uploads'
app.secret_key="hello"
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/",methods=['POST'])
def detect():
    if request.files.get('imageInput'):
        file=request.files['imageInput']
        if file:
            filename=secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))

        # image_path = 'C:/Users/Joyce Merin/PycharmProjects/BuildingDetection/23578960_15.png'
        img=cv2.imread('uploads/'+filename,1)
        # img=cv2.imread(image_path,1)
        # img=img_to_array(img)
        print(img.shape)


        SIZE_X=(img.shape[1]//patch_size)*patch_size
        SIZE_Y=(img.shape[0]//patch_size)*patch_size
        large_img=Image.fromarray(img)
        # large_img=Image.fromarray(img.astype('uint8'))
        large_img=large_img.crop((0,0,SIZE_X,SIZE_Y))

        large_img=np.array(large_img)
        print(large_img.shape)

        patches_img=patchify(large_img,(patch_size,patch_size,3),step=patch_size)
        patched_prediction=[]
        train_image_dataset=[]
        for i in range(patches_img.shape[0]):
            for j in range(patches_img.shape[1]):
                single_patch_img=patches_img[i,j,:,:]
                single_patch_img=scaler.fit_transform(single_patch_img.reshape(-1,single_patch_img.shape[-1])).reshape(single_patch_img.shape)
                single_patch_img=single_patch_img[0]
                single_patch_img=(single_patch_img*255).astype('uint8')
                single_patch_img=cv2.cvtColor(single_patch_img,cv2.COLOR_BGR2GRAY)
                single_patch_img=np.expand_dims(normalize(np.array(single_patch_img)),axis=0)
                pred=(pred_model.predict(single_patch_img)[0,:,:,0]>0.2).astype(np.uint8)
                patched_prediction.append(pred)

        patched_prediction=np.array(patched_prediction)
        patched_prediction=np.reshape(patched_prediction,[patches_img.shape[0],patches_img.shape[1],patch_size,patch_size])
        unpatched_prediction=unpatchify(patched_prediction,(large_img.shape[0],large_img.shape[1]))
        unpatched_prediction = (unpatched_prediction * 255).astype('uint8')
        result_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        cv2.imwrite(result_file_path, unpatched_prediction)


        retval, buffer = cv2.imencode('.jpg', img)
        image_src = 'data:image/jpeg;base64,' + str(base64.b64encode(buffer))[2:-1]


        retval, buffer = cv2.imencode('.jpg', unpatched_prediction)
        result_img = 'data:image/jpeg;base64,' + str(base64.b64encode(buffer))[2:-1]

        return render_template("index.html", image_src=image_src, result_img=result_img)


if __name__ == "__main__":
    app.run()
