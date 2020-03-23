#!/usr/bin/env python
import numpy as np
import tensorflow as tf
from attention import attention 
from get_12ECG_features import get_12ECG_features
from sklearn.preprocessing import MultiLabelBinarizer

def run_12ECG_classifier(data,header_data,classes,model):

    data1 = list()
    num_classes = len(classes)
    current_label = np.zeros(num_classes, dtype=int)
    current_score = np.zeros(num_classes)

    # Use your classifier here to obtain a label and score for each class. 
    if data.shape[1]>=30000:
        ext = data[:,0:30000]
    elif data.shape[1]<30000:
        ext = np.zeros([12,30000])
        for i in range(0,11):
            ext[i][0:len(data[i])]= data[i]

    data1.append(ext.T)
    data1 = np.stack(data1,axis=0)
            
    current_l = model.predict(data1)
    label = np.argmax(current_l, axis=1)
    current_label[label] = 1    

    score = model.predict_proba(data1)

    for i in range(num_classes):
        current_score[i] = np.array(score[0][i])

    return current_label, current_score

def load_12ECG_model():
    # # load json and create model
    # model_dir ='C:/Users/Asus/Documents/Python Scripts/Research/ECG/Models/'
    # json_file = open(model_dir+'model1.json', 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    # # load weights into new model
    # loaded_model.load_weights(model_dir+"model1.h5")

    ## Load attention model
    custom_ob = {'attention':attention}
    loaded_model = tf.keras.models.load_model('model-29-0.881622-1.3492.h5', custom_objects=custom_ob)
    return loaded_model