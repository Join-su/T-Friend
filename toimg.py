import pandas as pd
import text_to_image as text2img
import os
from compare import comp
import re
import numpy as np

from PIL import Image
from sklearn.preprocessing import LabelEncoder

'''raw_DATA2 = 'e_bill_2019.xlsx'
excel_PATH2 = 'C:\\Users\\ialab\\Desktop\\T_Friend_data\\업종코드_매칭\\'
img_PATH = "C:\\Users\\ialab\\Desktop\\T_Friend_data\\img\\e_bill_2019_품명_img\\"


if not os.path.exists(img_PATH):
    os.mkdir(img_PATH)

WIDTH, HEIGHT = 28, 28
train, test = 100000, 100000'''

#img_PATH = "C:\\Users\\ialab\\Desktop\\T-Friend\\img\\"

class img(object):
    def __init__(self,T, df,comend, excel_PATH, img_PATH):
        self.T = T
        self.df = df
        self.comend = comend
        self.excel_PATH = excel_PATH
        self.img_PATH = img_PATH

    def pre_img(self):

        if os.path.exists(self.img_PATH):
            for file in os.scandir(self.img_PATH):
                os.remove(file.path)

        WIDTH, HEIGHT = 28, 28
        train, test = 100000, 100000

        ###### 등록번호 '-' 기호 지우기 ####
        '''
        if self.T == '영수증':
            buyer = '회사등록번호'
        else:
            buyer = '사업자등록번호'
        '''
        buyer = 'NO_BIZ'
        seller = 'NO_BIZ_C'


        try:
            buyer_raw_data = self.df[buyer].str.split('-', n=2, expand=True)
            buyer_raw_data[buyer] = buyer_raw_data[0].str.cat(buyer_raw_data[1])
            buyer_raw_data[buyer] = buyer_raw_data[buyer].str.cat(buyer_raw_data[2]).copy()
            del (buyer_raw_data[0])
            del (buyer_raw_data[1])
            del (buyer_raw_data[2])
            buyer_raw_data = buyer_raw_data.astype('str')

            del (self.df[buyer])
            self.df[buyer] = buyer_raw_data[buyer].astype('str')
        except AttributeError:
            print('-부호 없음')

        try:
            buyer_raw_data2 = self.df[seller].str.split('-', n=2, expand=True)
            print('buyer_raw_data2', buyer_raw_data2)
            buyer_raw_data2[seller] = buyer_raw_data2[0].str.cat(buyer_raw_data2[1])
            buyer_raw_data2[seller] = buyer_raw_data2[seller].str.cat(buyer_raw_data2[2]).copy()
            del (buyer_raw_data2[0])
            del (buyer_raw_data2[1])
            del (buyer_raw_data2[2])
            buyer_raw_data2 = buyer_raw_data2.astype('str')

            del (self.df['NO_BIZ_C'])
            self.df['NO_BIZ_C'] = buyer_raw_data2['NO_BIZ_C'].astype('str')

            ###################################
        except AttributeError:
            print('-부호 없음')




        # print(ind[0])
        name = buyer
        # xl = excel_PATH2 + 'total_17_18_new.xlsx'
        name2 = 'NO_BIZ_C'
        target = 'CD_ACCOUNT'
        if self.T == '계산서' :
            e_name = 'e_bill_2019_uniq.json'
        elif self.T == '영수증':
            e_name = 'cash_train.json'
        elif self.T == '기타' :
            e_name = 'etc.json'
        df = comp(self.comend,self.excel_PATH, self.T, self.df, target, e_name, name, name2)

        # df = comp(df,target, 'e_bill_2019_uniq.xlsx',name)

        # print(len(data))
        # print(df)
        print(df.head())

        if self.T == '영수증' or self.T == '기타':
            pre_data = df.loc[:, [name,name2]].astype('str')
            pre_data[name] = pre_data[name].str.replace(' ','')
            pre_data[name2] = pre_data[name2].str.replace(' ','')
            if self.comend == 'train':
                pre_data[target] = df[target].astype('str')
        else :
            name = 'NM_ITEM'
            pre_data = df.loc[:, [name]].astype('str')
            pre_data[name] = pre_data[name].str.replace(' ', '')
            if self.comend == 'train':
                pre_data[target] = df[target].astype('str')
            #print(pre_data.head())

        if os.path.exists(self.img_PATH):
            print("already eixts path")
        else :
            os.mkdir(self.img_PATH)
            print("create path")

        for i in range(len(pre_data)):
            img_num = train + i
            obj = pre_data.loc[i, [name]].item()
            # obj += pre_data.loc[i,[name2]].item() ##name2도 이미지화 하는데 같이 고려해야 한다면
            if self.comend == 'train':
                c_num = pre_data.loc[i, [target]].str.split(' ', n=2, expand=True)
                text2img.encode(obj, self.img_PATH + 'a_%d_%d.png' % (img_num, c_num[0]))
                img = Image.open(self.img_PATH + 'a_%d_%d.png' % (img_num, c_num[0]))
                resize_img = img.resize((WIDTH, HEIGHT))
                resize_img.save(self.img_PATH + 'a_%d_%d.png' % (img_num, c_num[0]))
                print('이미지화 : %d / %d' % (i, len(pre_data)))
            else :
                #c_num = pre_data.loc[i, [target]].str.split(' ', n=2, expand=True)
                text2img.encode(obj, self.img_PATH + 'a_%d.png' % (img_num))
                img = Image.open(self.img_PATH + 'a_%d.png' % (img_num))
                resize_img = img.resize((WIDTH, HEIGHT))
                resize_img.save(self.img_PATH + 'a_%d.png' % (img_num))
                print('이미지화 : %d / %d' % (i, len(pre_data)))




