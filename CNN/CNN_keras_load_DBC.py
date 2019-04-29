from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation
from PIL import Image
import win32com.client

import numpy as np
import pandas as pd
import os
import csv
import tensorflow as tf
import matplotlib.pyplot as plt


d18_train = 'train_test18.xlsx'
d17_train = 'train_test17.xlsx'
excel_PATH = "C:\\Git\\T-Friend\\3friend_raw_data\\"
img_PATH = "C:\\Git\\T-Friend\\img_data_combine\\"
labels_val = []

def dataset(images):
    #data = pd.read_csv(PATH, header=None)
    #images = data.iloc[:, :].values
    images = images.astype(np.float)
    images = images.reshape(28, 28, 1)
    images = np.multiply(images, 1.0 / 255.0)
    return images


def data_set_fun(path, set_size, label_no=0):

    count = 0
    filename_list = os.listdir(path)

    if label_no !=0 :
        path = path + '\\' + str(label_no)
        filename_list = os.listdir(path)
    if set_size == 0 :
        set_size = len(filename_list)

    print(filename_list)
    print(len(filename_list))
    X_set = np.empty((set_size, 28, 28, 1), dtype=np.float32)
    Y_set = np.empty((set_size), dtype=np.float32)

    name = []

    #np.random.shuffle(filename_list)
    #result = dict()

    for i, filename in enumerate(filename_list):
        if i >= set_size :
            break
        #name.append(filename)
        label = filename.split('.')[0]
        label = label.split('_')[2]
        #result[label] = result.setdefault(label,0)+1
        #print("label",label)
        Y_set[i] = int(label)


        file_path = os.path.join(path, filename)
        img = Image.open(file_path)
        imgarray = np.array(img)
        imgarray = imgarray.flatten()
        #print(imgarray)

        images = dataset(imgarray)
        X_set[i] = images

        labels_val.append(int(label))

    #if train:
    #    return X_set, Y_set, result
    return X_set, Y_set




def dence_to_one_hot(labels_dence, num_classes):
    #print('labels_dence : ',labels_dence)
    num_labes = labels_dence.shape[0]
    #print('num_labes : ',num_labes)
    index_offset = np.arange(num_labes) * num_classes
    #print('index_offset : ',index_offset)
    labels_one_hot = np.zeros((num_labes, num_classes))
    #print('labels_dence.ravel() : ',labels_dence.ravel())
    labels_one_hot.flat[index_offset + labels_dence.ravel()] = 1 #flat - 배열을 1차원으로 두고 인덱스를 이용해 값 확인
    #print('llabels_one_hot : ', labels_one_hot)
    return labels_one_hot

def index_label(label):
    #print(label)
    list = []
    for j in range(len(label)):
        for i in range(len(labels_val)):
            if int(label[j]) == int(labels_val[i]):
                list.append(i)
                break
    return np.asarray(list)



test_labels = [146, 150, 166, 202, 401, 413, 809, 810, 811, 812, 813, 814, 817, 819, 820, 823, 824, 826, 827, 828, 831]
#labels_val = [131, 146, 150, 166, 172, 192, 194, 196, 198, 202, 209, 211, 246, 267, 387, 401, 404, 412, 413, 615, 715, 809, 810, 811, 812, 813, 814, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 831]
labels_val = [131, 141, 146, 150, 166, 172, 192, 194, 196, 198, 202, 209, 211, 246, 254, 260, 267, 387, 401, 404, 412, 615, 715, 727, 809, 812, 813, 814, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 831]

labels_val.sort()
print(labels_val)
labels_count = len(labels_val)
print('labels_val : ',labels_val)
count_epoch = 1
la = 146
# trX, trY, result = data_set_fun(Img_Path + 'train_img', 0, 0)
# print('train_분포 : ', result)
teX, teY = data_set_fun(img_PATH + 'img_18_2', 0, 0)
# print('test_분포 : ', result)
# print('tset 분포 :', len(teX))

# labels_val = list(set(labels_val))

# trY = index_label(trY)
teY = np_utils.to_categorical(teY, labels_count)

# print(len(teY), len(trY))
# trY = dence_to_one_hot(trY, labels_count)
#teY = dence_to_one_hot(teY, labels_count)

loss = []
acc = []

#


model = load_model('./model_category_2/model_category_17_18')

predict = model.predict(teX)
predict = predict.argmax(axis = -1)
print(predict)
for i,pre in enumerate(predict):
    #print(pre)
    print(labels_val[pre])




    #model = load_model('./model_ResNet/model_num-%s.h5' % num)


emnist1_acc = model.evaluate(teX, teY)
e1_acc = emnist1_acc[1]
print("\nAcc: %.4f" % e1_acc)

print("\nTest score:", emnist1_acc[0])
print('Test accuracy:', emnist1_acc[1])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
