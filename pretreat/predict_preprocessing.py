import pandas as pd
import text_to_image as text2img
import os
import numpy as np

from PIL import Image
from sklearn.preprocessing import LabelEncoder, LabelBinarizer

data_PATH = 'C:\\Git\\T-Friend\\predict_excel\\'
img_PATH = "C:\\Git\\T-Friend\\img_predict_data\\"
if not os.path.exists(img_PATH):
    os.mkdir(img_PATH)

predict_FILE = 'credit_cash_predict.xlsx'

WIDTH, HEIGHT = 28, 28
train, test = 100000, 100000

raw_data = pd.read_excel(data_PATH + predict_FILE, encoding='utf-8')
pre_data = raw_data.loc[:, ['회사명', '가맹점명', 'c']]
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
pre_data['거래처'] = pre_data['가맹점명'].astype('str')
pre_data['obj'] = pre_data['회사명'].str.cat(pre_data['가맹점명'])
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

for i in range(len(pre_data)):
    img_num = train + i
    obj = obj_list[i]
    c_num = 100
    text2img.encode(obj, img_PATH + 'a_%d_%d.png' %(img_num, c_num))
    img = Image.open(img_PATH + 'a_%d_%d.png' %(img_num, c_num))
    resize_img = img.resize((WIDTH, HEIGHT))
    resize_img.save(img_PATH + 'a_%d_%d.png' %(img_num, c_num))
    print('%d / %d' %(i, len(pre_data)))
