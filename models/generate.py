import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import json
def readDataSet():
    imgs = np.empty((0,50,50))
    imgs_tags = []
    path = "./age_dataset"
    valid_images = [".jpg",".gif",".png",".tga"]
    i = 0
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if int(f.split('_')[0]) > 18:
            imgs_tag = 1
        else:
            imgs_tag = 0
        if ext.lower() not in valid_images:
            continue
        if i == 5:
            break
        img = Image.open(os.path.join(path,f))
        img = np.reshape(img.resize([50, 50]).convert('L'), (1, 50, 50))
        imgs = np.append(imgs, np.array(img), axis=0)
        imgs_tags = np.append(imgs_tags, imgs_tag)
        i += 1
    return imgs, imgs_tags

imgs, _ = readDataSet()
# %% 
im = imgs.reshape(imgs.shape[0], 50, 50, 1)
img = im[0].reshape(50, 50)
img = Image.fromarray(img).convert('L')

#%%
with open("models/in.json", "w") as f:
    json.dump({'in': im[0].tolist()}, f)

f = open('models/in.json')
data = json.load(f)
dataNp = np.array(data['in'])