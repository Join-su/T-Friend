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
img_PATH = "C:\\Git\\T-Friend\\img_predict_data"
predict_PATH = "C:\\Git\\T-Friend\\predict_excel\\"
if not os.path.exists(predict_PATH):
    os.mkdir(predict_PATH)

labels_val = []


def dataset(images):
    #data = pd.read_csv(PATH, header=None)
    #images = data.iloc[:, :].values
    images = images.astype(np.float)
    images = images.reshape(28, 28, 1)
    images = np.multiply(images, 1.0 / 255.0)
    return images


def data_set_fun(path, set_size, label_no=0):

    filename_list = os.listdir(path)

    if label_no !=0 :
        path = path + '\\' + str(label_no)
        filename_list = os.listdir(path)
    if set_size == 0 :
        set_size = len(filename_list)

    X_set = np.empty((set_size, 28, 28, 1), dtype=np.float32)
    Y_set = np.empty((set_size), dtype=np.float32)

    for i, filename in enumerate(filename_list):
        if i >= set_size :
            break
        label = filename.split('.')[0]
        label = label.split('_')[2]
        Y_set[i] = int(label)

        file_path = os.path.join(path, filename)
        img = Image.open(file_path)
        imgarray = np.array(img)
        imgarray = imgarray.flatten()

        images = dataset(imgarray)
        X_set[i] = images

        labels_val.append(int(label))

    return X_set, Y_set


labels_val = list(set(labels_val))
labels_val.sort()
print(labels_val)
labels_count = len(labels_val)

teX, teY = data_set_fun(img_PATH, 0, 0)

MODEL_DIR = "C:\\Git\\T-Friend\\model_save\\"
FILE_NAME = 'model_complete.h5'

model = load_model(MODEL_DIR + FILE_NAME)

predict = model.predict_classes(teX)

df = pd.DataFrame(predict)
df.to_excel(predict_PATH + "predict_test.xlsx")

print(predict)
