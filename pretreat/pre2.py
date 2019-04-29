import pandas as pd
import text_to_image as text2img
import os
import numpy as np

from PIL import Image
from sklearn.preprocessing import LabelEncoder


d18_train = 'train_test18.xlsx'
d17_train = 'train_test17.xlsx'
raw_DATA = '3friend_raw_data.xlsx'
excel_PATH = "C:\\Git\\T-Friend\\3friend_raw_data\\"
img_PATH = "C:\\Git\\T-Friend\\img_data_predict\\"
predict_PATH = "C:\\Git\\T-Friend\\predict_excel\\"
predict_FILE = 'credit_cash_predict.xlsx'

if not os.path.exists(img_PATH):
    os.mkdir(img_PATH)

WIDTH, HEIGHT = 28, 28
train, test = 100000, 100000

credit_raw_data = pd.read_excel(excel_PATH + raw_DATA, sheet_name='신용카드(매입)', header=2, usecols="C:R", encoding='utf-8')
credit_raw_data = credit_raw_data.drop(0, 0)
cash_raw_data = pd.read_excel(excel_PATH + raw_DATA, sheet_name='현금영수증', header=2, usecols='C:P', encoding='utf-8')
cash_raw_data = cash_raw_data.drop(0, 0)

credit_data = credit_raw_data.loc[:, ['회사명', '가맹점명']].copy()
cash_data = cash_raw_data.loc[:, ['회사명', '가맹점명']].copy()
data = pd.concat([credit_data, cash_data], ignore_index=True)
data.to_excel(predict_PATH + predict_FILE)
pre_data = data.loc[:, ['회사명', '가맹점명']]
pre_data['회사명'] = pre_data['회사명'].astype('str')
pre_data['가맹점명'] = pre_data['가맹점명'].astype('str')
pre_data['obj'] = pre_data['회사명'].str.cat(pre_data['가맹점명'])
pre_data['obj'] = pre_data['obj'].astype('str')
obj_list = pre_data['obj']

for i in range(len(pre_data)):
    img_num = train + i
    obj = obj_list[i]
    c_num = 11
    text2img.encode(obj, img_PATH + 'a_%d_%d.png' %(img_num, c_num))
    img = Image.open(img_PATH + 'a_%d_%d.png' %(img_num, c_num))
    resize_img = img.resize((WIDTH, HEIGHT))
    resize_img.save(img_PATH + 'a_%d_%d.png' %(img_num, c_num))
    print('%d / %d' %(i, len(pre_data)))

