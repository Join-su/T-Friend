import pandas as pd
import text_to_image as text2img
import os
import numpy as np

from PIL import Image
from sklearn.preprocessing import LabelEncoder, LabelBinarizer


d18_train = 'train_test18.xlsx'
d17_train = 'train_test17.xlsx'
excel_PATH = "C:\\Git\\T-Friend\\3friend_raw_data\\"
img_PATH = "C:\\Git\\T-Friend\\new_train_img_data11\\"
if not os.path.exists(img_PATH):
    os.mkdir(img_PATH)

WIDTH, HEIGHT = 28, 28
train, test = 100000, 100000

raw_data_18 = pd.read_excel(excel_PATH + d18_train, encoding='utf-8')
raw_data_17 = pd.read_excel(excel_PATH + d17_train, encoding='utf-8')
data = pd.concat([raw_data_17, raw_data_18], ignore_index=True)
pre_data = data.loc[:, ['회사명', '거래처', 'c', '계정과목']]
lists = pre_data['c']
index = np.unique(lists)
index_num = LabelEncoder().fit_transform(index)
pre_data['c'] = LabelEncoder().fit_transform(pre_data['c'])
num_list = pre_data['c']
C = pd.DataFrame(LabelBinarizer().fit_transform(pre_data['c']), columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7']
                 , index=pre_data.index)
del(pre_data['c'])
C = C.astype('str')
pre_data['회사명'] = pre_data['회사명'].astype('str')
pre_data['거래처'] = pre_data['거래처'].astype('str')
pre_data['obj'] = pre_data['회사명'].str.cat(pre_data['거래처'])
pre_data['obj'] = pre_data['obj'].astype('str')
pre_data['obj'] = pre_data['obj'].str.cat(C['c0'])
pre_data['obj'] = pre_data['obj'].str.cat(C['c1'])
pre_data['obj'] = pre_data['obj'].str.cat(C['c2'])
pre_data['obj'] = pre_data['obj'].str.cat(C['c3'])
pre_data['obj'] = pre_data['obj'].str.cat(C['c4'])
pre_data['obj'] = pre_data['obj'].str.cat(C['c5'])
pre_data['obj'] = pre_data['obj'].str.cat(C['c6'])
pre_data['obj'] = pre_data['obj'].str.cat(C['c7'])
obj_list = pre_data['obj']

label = pre_data.loc[:, ['계정과목']].copy()
label_index = label['계정과목'].str.split(' ', n=1, expand=True)
label_list = list(label_index)
Y = label_index[0].copy()
Y = Y.astype('int')

for i in range(len(pre_data)):
    img_num = train + i
    obj = obj_list[i]
    c_num = Y[i]
    text2img.encode(obj, img_PATH + 'a_%d_%d.png' %(img_num, c_num))
    img = Image.open(img_PATH + 'a_%d_%d.png' %(img_num, c_num))
    resize_img = img.resize((WIDTH, HEIGHT))
    resize_img.save(img_PATH + 'a_%d_%d.png' %(img_num, c_num))
    print('%d / %d' %(i, len(pre_data)))
