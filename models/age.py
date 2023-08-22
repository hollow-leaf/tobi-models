# %% Import libraries
# TensorFlow and tf.keras
import json
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Lambda,
    Conv2D,
    AveragePooling2D,
    MaxPooling2D,
    Softmax,
    BatchNormalization,
    GlobalAveragePooling2D
)
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
# Helper libraries
import os, os.path
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

print('tensorflow version', tf.__version__)

# %% Read Picture
def readDataSet():
    imgs = np.empty((0,50,50))
    imgs_tags = []
    path = "./age_dataset"
    valid_images = [".jpg",".gif",".png",".tga"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if int(f.split('_')[0]) > 18:
            imgs_tag = 1
        else:
            imgs_tag = 0
        if ext.lower() not in valid_images:
            continue
        img = Image.open(os.path.join(path,f))
        img = np.reshape(img.resize([50, 50]).convert('L'), (1, 50, 50))
        imgs = np.append(imgs, np.array(img), axis=0)
        imgs_tags = np.append(imgs_tags, imgs_tag)
    print(imgs_tags)
    return imgs, imgs_tags
# %%
inputs = Input(shape=(50,50,1))
out = Conv2D(4, 3, use_bias=False)(inputs)
out = BatchNormalization()(out)
out = AveragePooling2D()(out)
out = GlobalAveragePooling2D()(out) # best practice: use GlobalAveragePooling2D instead of Flatten
out = Dense(2, activation=None)(out)
out = Softmax()(out)
model = Model(inputs, out)
model.summary()
# %%
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['acc']
    )

x, y = readDataSet()
print(type(x), x.shape)
print(type(y), y.shape)
y = to_categorical(y)
data_train, data_test, labels_train, labels_test = train_test_split(
    x, 
    y, 
    test_size=0.20, 
    random_state=42)
data_train = data_train.reshape(data_train.shape[0], 50, 50, 1)
data_test = data_test.reshape(data_test.shape[0], 50, 50, 1)
print(data_train.shape)
print(labels_train.shape)
# %%

model.fit(data_train, 
          labels_train, 
          epochs=100, 
          batch_size=128, 
          validation_data=(data_test, labels_test))

model.save('models/age_garbage.h5')

#%%
model2 = Model(model.input, model.layers[-2].output)
model2.layers[-1]

X = data_test[[0]]
y = model2.predict(X)
y

for layer in model.layers:
    print(layer.__class__.__name__, layer.get_config())
    try:
        print(layer.get_config()['function'])
    except:
        pass
    print(layer.get_input_shape_at(0),layer.get_output_shape_at(0))
    try:
        print(layer.get_weights()[0].shape)
        print(layer.get_weights()[1].shape)
    except:
        pass
#%%
with open("models/age_garbage.json", "w") as f:
    json.dump({'X': X.flatten().tolist(), 'y': y.flatten().tolist()}, f)