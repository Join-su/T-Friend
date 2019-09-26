#import load_cnn
#import load_rf
from load_cnn import *
from load_rf import *
from RF_train import *
from toimg import *
from filter_cnn import *
from filter_rf import *
import pandas as pd
import os

### 파일 불러오기(pandas)
PATH = "C:\\Users\\ialab\\Desktop\\T-Friend\\"
excel_PATH = 'C:\\Users\\ialab\\Desktop\\T-Friend\\pre\\'
img_PATH = "C:\\Users\\ialab\\Desktop\\T-Friend\\img\\"

filename = '34_file.xlsx' ## 영수증
filename2 = 'e_bill_2019.xlsx'  ## 계산서
#filename3 = 'total_17_18.xlsx'
filename3 = 'e_train_data.xlsx'
#comend = 'test'
comend = 'test'

img_full_name = PATH + filename
df = pd.read_excel(img_full_name, encoding='utf-8', index_col=0)

### fearture에 품명이 있는지 검사(있으면 계산서, 없으면 영수증 취급)
filename_spl = filename.split('_')
#col_list = df.columns.tolist()
#T = '영수증'

T = '영수증'
if filename_spl[0] == '12': T = '계산서'


#for i in range(len(col_list)):
#    if col_list[i] == '품명' : T = '계산서'

print(df.head())
### 이미지화
a = img(T,df,comend,excel_PATH, img_PATH)
a.pre_img()



### 해당 종류의 CNN 불러와서 C37 예측
b = CNN(img_PATH, T, excel_PATH, img_full_name, comend)
b.main_cnn()



### CNN정확도 낮은 것 따로 빼놓음
c = FILTER_CNN(T,excel_PATH)
c.main_f_cnn()



if comend == 'train':
    ### 해당 종류의 RF 불러와서 결과 예측
    d1 = RF_train(T, excel_PATH)
    d1.main_RF_train()
else :
    ### 해당 종류의 RF 불러와서 결과 예측
    d2 = RF(T,excel_PATH)
    d2.main_RF()


### RF정화도가 낮은 것 따로 빼놓음
e = FILTER_RF(T,excel_PATH)
e.main_f_rf()

