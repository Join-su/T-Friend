from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation
from PIL import Image
#import win32com.client

import numpy as np
import pandas as pd
import os
import csv
import tensorflow as tf
import matplotlib.pyplot as plt

'''
d18_train = 'train_test18.xlsx'
d17_train = 'train_test17.xlsx'
raw_DATA = 'e_bill_2019_uniq.xlsx'
excel_PATH = "C:\\Users\\ialab\\Desktop\\T_Friend_data\\T-Friend\\3friend_raw_data\\"
excel_PATH2 = 'C:\\Users\\ialab\\Desktop\\T_Friend_data\\업종코드_매칭\\'
img_PATH = "C:\\Users\\ialab\\Desktop\\T_Friend_data\\img\\e_bill_2019_품명_img\\"
predict_PATH = "C:\\Git\\T-Friend\\predict_excel\\"
predict_FILE = 'tarin_industry_code.xlsx'
'''
labels_val = [141, 146, 150, 166, 172, 192, 194, 196, 198, 202, 209, 211, 254, 260, 267, 387, 401, 615, 715,
                      727, 809, 812, 813, 814, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 831]

class CNN(object):
    def __init__(self, img_PATH ,T, excel_PATH,img_full_name, comend):
        self.img_PATH = img_PATH
        self.T = T
        self.excel_PATH = excel_PATH
        self.comend = comend
        self.img_full_name = img_full_name

    def dataset(self,images):
        # data = pd.read_csv(PATH, header=None)
        # images = data.iloc[:, :].values
        images = images.astype(np.float)
        images = images.reshape(28, 28, 1)
        images = np.multiply(images, 1.0 / 255.0)
        return images

    def data_set_fun(self,path, set_size, label_no=0):

        count = 0
        filename_list = os.listdir(path)

        if label_no != 0:
            path = path + '\\' + str(label_no)
            filename_list = os.listdir(path)
        if set_size == 0:
            set_size = len(filename_list)

       # print(filename_list)
       # print(len(filename_list))
        X_set = np.empty((set_size, 28, 28, 1), dtype=np.float32)
        Y_set = np.empty((set_size), dtype=np.float32)

        name = []

        # np.random.shuffle(filename_list)
        # result = dict()

        for i, filename in enumerate(filename_list):
            if i >= set_size:
                break
            # name.append(filename)
            if self.comend == 'train' :
                label = filename.split('.')[0]
                label = label.split('_')[2]
                # result[label] = result.setdefault(label,0)+1
                # print("label",label)
                Y_set[i] = int(label)
                # print(i)

            file_path = os.path.join(path, filename)
            #print('file_path', file_path)
            img = Image.open(file_path)
            imgarray = np.array(img)
            imgarray = imgarray.flatten()
            #print(np.shape(imgarray))

            images = self.dataset(imgarray)
            #print(np.shape(images))
            X_set[i] = images

            # labels_val.append(int(label))

        # if train:
        #    return X_set, Y_set, result
        if self.comend == 'train':
            return X_set, Y_set
        return X_set

    def dence_to_one_hot(self,labels_dence, num_classes):
        #print('labels_dence : ',labels_dence)
        num_labes = labels_dence.shape[0]
        #print('num_labes : ',num_labes)
        index_offset = np.arange(num_labes) * num_classes
        #print('index_offset : ',index_offset)
        labels_one_hot = np.zeros((num_labes, num_classes))
        #print('labels_dence.ravel() : ',labels_dence.ravel())
        labels_one_hot.flat[index_offset + labels_dence.ravel()] = 1  # flat - 배열을 1차원으로 두고 인덱스를 이용해 값 확인
        #print('llabels_one_hot : ', labels_one_hot)
        return labels_one_hot

    def index_label(self,label):
        # print(label)
        list = []
        for j in range(len(label)):
            for i in range(len(labels_val)):
                if int(label[j]) == int(labels_val[i]):
                    list.append(i)
                    break
        return np.asarray(list)

    def main_cnn(self):

        labels_val = [141, 146, 150, 166, 172, 192, 194, 196, 198, 202, 209, 211, 254, 260, 267, 387, 401, 615, 715,
                      727, 809, 812, 813, 814, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 831]

        if self.comend == 'train':

            trX, trY = self.data_set_fun(self.img_PATH, 0, 0)  # train_img_new
            # print('train_분포 : ', result)
            #teX, teY, result = self.data_set_fun(img_test_PATH, 0, 0)  # test_img_new
            # print('test_분포 : ', result)

            labels_val = list(set(labels_val))
            labels_val.sort()
            print(labels_val)
            print('labels_val len : ', len(labels_val))
            labels_count = len(labels_val)

            trY = self.index_label(trY)
            #teY = self.index_label(teY)

            # print(len(teY), len(trY))
            trY = self.dence_to_one_hot(trY, labels_count)
            #teY = self.dence_to_one_hot(teY, labels_count)

            if self.T == '계산서': EPOCH = 400
            else : EPOCH = 800
            BATCH_SIZE = 128
            VERBOSE = 2

            model = Sequential()
            model.add(
                Conv2D(20, kernel_size=5, input_shape=(28, 28, 1), padding="same", kernel_initializer='he_uniform'))
            # model.add(BatchNormalization(axis=1))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
            model.add(Conv2D(50, kernel_size=5, padding="same", kernel_initializer='he_uniform'))
            # model.add(BatchNormalization(axis=1))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
            model.add(Dropout(0.2))
            model.add(Flatten())
            model.add(Dense(500, activation='relu', kernel_initializer='he_uniform'))
            model.add(Dropout(0.3))
            model.add(Dense(labels_count, activation='softmax'))

            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            MODEL_DIR = "/home/cent/Documents/github/save_file/"
            if not os.path.exists(MODEL_DIR):
                os.mkdir(MODEL_DIR)

            '''
            checkpoint = ModelCheckpoint(FILE_NAME)
            callbacks_list = [checkpoint]
            '''
            history = model.fit(trX, trY, epochs=EPOCH, batch_size=BATCH_SIZE, shuffle=True, verbose=VERBOSE)

            if self.T =='계산서' :
                FILE_NAME = 'categori_update_new.h5'
            else :
                FILE_NAME = 'categori_update_cash.h5'
            model.save(MODEL_DIR + FILE_NAME)

            #emnist1_acc = model.evaluate(teX, teY)
            #e1_acc = emnist1_acc[1]
            #print("\nAcc: %.4f" % e1_acc)
            #print(result)

            #print("\nTest score:", emnist1_acc[0])
            #print('Test accuracy:', emnist1_acc[1])

            print(history.history.keys())

        else :

            labels_val = list(set(labels_val))
            labels_val.sort()
            print(labels_val)
            labels_count = len(labels_val)
            print('labels_val : ', labels_val)
            count_epoch = 1
            la = 146
            # trX, trY, result = data_set_fun(Img_Path + 'train_img', 0, 0)
            # print('train_분포 : ', result)
            teX = self.data_set_fun(self.img_PATH, 0, 0)
            # print('test_분포 : ', result)
            # print('tset 분포 :', len(teX))

            # labels_val = list(set(labels_val))

            #print('teY : ',teY)

            #teY_1 = self.index_label(teY)
            #print('tey_1 : ', teY_1)
            #print('y_len : ', len(list(set(teY))))

            # teY = np_utils.to_categorical(teY, len(labels_val))

            # print(len(teY), len(trY))
            # trY = dence_to_one_hot(trY, labels_count)
            #teY = self.dence_to_one_hot(teY_1, labels_count)

            loss = []
            acc = []

            #
            if self.T == '계산서':
                model = load_model('/home/cent/Documents/github/save_file/categori_update_new.h5')
            else :
                model = load_model('/home/cent/Documents/github/save_file/categori_update_cash.h5')

            '''
            emnist1_acc = model.evaluate(teX, teY)
            e1_acc = emnist1_acc[1]
            print("\nAcc: %.4f" % e1_acc)
    
            print("\nTest score:", emnist1_acc[0])
            print('Test accuracy:', emnist1_acc[1])
            '''
            pred = model.predict(teX)
            predict = pred.argmax(axis=-1)
            '''
            print('predict : ', pred)
            print('predict : ', predict)
            for i in range(len(pred)):
                print('result : ', pred[i][predict[i]])
            '''
            count = 0
            # print(predict)
            if self.T == '계산서' :
                raw_DATA = 'e_bill_2019_uniq.xlsx'

            elif self.T == '영수증':
                raw_DATA = 'cash_train.xlsx'

            elif self.T == '기타' :
                raw_DATA = 'etc.xlsx'

            data = pd.read_excel(self.img_full_name, encoding='utf-8', index_col=0)
            data['cc'] = 0
            data['predict'] = 0
            for i, pre in enumerate(predict):
                # print(pre)
                data.loc[i, ['cc']] = labels_val[pre]
                data.loc[i, ['predict']] = pred[i][pre]
                print("카테고리 설정 : %d/%d" % (i, len(predict)))

                # print(labels_val[pre])

            data.to_excel(self.excel_PATH + raw_DATA)

            # model = load_model('./model_ResNet/model_num-%s.h5' % num)


