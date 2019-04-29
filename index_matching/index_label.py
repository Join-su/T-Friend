from keras.utils import np_utils
from keras.models import load_model
from PIL import Image

import numpy as np
import pandas as pd
import os
import text_to_image as text2img


raw_DATA = '3friend_raw_data.xlsx'
d18_train = 'train_test18.xlsx'
d17_train = 'train_test17.xlsx'
excel_PATH = "C:\\Git\\T-Friend\\3friend_raw_data\\"
img_PATH = "C:\\Git\\T-Friend\\img_data_predict"
img2_PATH = "C:\\Git\\T-Friend\\new_train_img_data"
predict_PATH = "C:\\Git\\T-Friend\\predict_excel\\"

labels_val = []


def label_list(path):

    filename_list = os.listdir(path)
    set_size = len(filename_list)
    Y_set = np.empty((set_size), dtype=np.float32)

    for i, filename in enumerate(filename_list):

        label = filename.split('.')[0]
        label = label.split('_')[2]
        Y_set[i] = int(label)

        labels_val.append(int(label))

    return Y_set


def dataset(images):
    #data = pd.read_csv(PATH, header=None)
    #images = data.iloc[:, :].values
    images = images.astype(np.float)
    images = images.reshape(28, 28, 1)
    images = np.multiply(images, 1.0 / 255.0)
    return images


def data_set_fun(path):

    filename_list = os.listdir(path)
    set_size = len(filename_list)

    X_set = np.empty((set_size, 28, 28, 1), dtype=np.float32)

    for i, filename in enumerate(filename_list):

        file_path = os.path.join(path, filename)
        img = Image.open(file_path)
        imgarray = np.array(img)
        imgarray = imgarray.flatten()

        images = dataset(imgarray)
        X_set[i] = images

    return X_set


teX = data_set_fun(img_PATH)

MODEL_DIR = "C:\\Git\\T-Friend\\model_save\\"
FILE_NAME = 'model_complete.h5'

model = load_model(MODEL_DIR + FILE_NAME)

predict = model.predict_classes(teX)

label = label_list(img2_PATH)

labels_val = list(set(labels_val))
labels_val.sort()
print(labels_val)
labels_count = len(labels_val)

predicted_label = []

for i, pre in enumerate(predict):
    predicted_label.append(labels_val[pre])

df = pd.DataFrame(predicted_label)
df.to_excel(predict_PATH + "predict_final_test.xlsx")

