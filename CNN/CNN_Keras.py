from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation
from keras.callbacks import ModelCheckpoint
from PIL import Image

import numpy as np
import pandas as pd
import os
import csv
import tensorflow as tf

import matplotlib.pyplot as plt

Img_Path = "C:\\Users\\ialab\\PycharmProjects\\insu_CNN\\"
labels_val = []

def dataset(images):
    #data = pd.read_csv(PATH, header=None)
    #images = data.iloc[:, :].values
    images = images.astype(np.float)
    images = images.reshape(28, 28, 1)
    images = np.multiply(images, 1.0 / 255.0)
    return images


def data_set_fun(path, set_size, label_no):

    count = 0
    filename_list = os.listdir(path)
    if set_size == 0 :
        set_size = len(filename_list)
    if label_no !=0:
        path = path + '\\' + str(label_no)
        filename_list = os.listdir(path)

    X_set = np.empty((set_size, 28, 28, 1), dtype=np.float32)
    Y_set = np.empty((set_size), dtype=np.float32)

    name = []

    np.random.shuffle(filename_list)
    result = dict()

    for i, filename in enumerate(filename_list):
        if i >= set_size :
            break
        #name.append(filename)
        label = filename.split('.')[0]
        label = label.split('_')[2]
        result[label] = result.setdefault(label,0)+1
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
    return X_set, Y_set, result



def dence_to_one_hot(labels_dence, num_classes):
    #print(labels_dence)
    num_labes = labels_dence.shape[0]
    #print(num_labes)
    index_offset = np.arange(num_labes) * num_classes
    #print(index_offset)
    labels_one_hot = np.zeros((num_labes, num_classes))
    #print(labels_dence.ravel())
    labels_one_hot.flat[index_offset + labels_dence.ravel()] = 1 #flat - 배열을 1차원으로 두고 인덱스를 이용해 값 확인
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

trX, trY, result = data_set_fun(Img_Path + 'img_17_18_2', 0, 0)#train_img_new
#print('train_분포 : ', result)
teX, teY, result = data_set_fun(Img_Path + 'img_18_2', 0, 0)#test_img_new
#print('test_분포 : ', result)

labels_val = list(set(labels_val))
labels_val.sort()
print(labels_val)
labels_count = len(labels_val)

trY = index_label(trY)
teY = index_label(teY)

#print(len(teY), len(trY))
trY = dence_to_one_hot(trY, labels_count)
teY = dence_to_one_hot(teY, labels_count)

EPOCH = 600
BATCH_SIZE = 128
VERBOSE = 2

model = Sequential()
model.add(Conv2D(20, kernel_size=5, input_shape=(28, 28, 1), padding="same", kernel_initializer = 'he_uniform'))
#model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(50, kernel_size=5, padding="same", kernel_initializer = 'he_uniform'))
#model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(500, activation='relu', kernel_initializer = 'he_uniform'))
model.add(Dropout(0.3))
model.add(Dense(labels_count, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

MODEL_DIR = './model_category_2/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

'''
checkpoint = ModelCheckpoint(FILE_NAME)
callbacks_list = [checkpoint]
'''
history = model.fit(trX, trY, validation_split=0.3,
                          epochs=EPOCH, batch_size=BATCH_SIZE, shuffle = True,verbose=VERBOSE)

FILE_NAME = './model_category_2/model_category_17_18'
model.save(FILE_NAME)

#model.save(FILE_NAME)
emnist1_acc = model.evaluate(teX, teY)
e1_acc = emnist1_acc[1]
print("\nAcc: %.4f" % e1_acc)
print(result)

print("\nTest score:", emnist1_acc[0])
print('Test accuracy:', emnist1_acc[1])

print(history.history.keys())

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



'''
reset_weights(model)

predict = model.predict(teX)
predict = predict.argmax(axis = -1)
print(predict)
for i,pre in enumerate(predict):
    #print(pre)
    print(labels_val[pre])
'''
