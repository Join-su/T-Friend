#-*- coding: utf-8 -*-

import numpy as np
import win32com.client
import text_to_image
#import input_data
from PIL import Image
import sys
import io
import os
import csv

#path = 'C:\\Users\\aiia\\.atom\\python\\text_to_image\\train'

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')

excel = win32com.client.Dispatch("Excel.Application")

excel_file = excel.Workbooks.Open('C:\\Users\\ialab\\PycharmProjects\\insu_CNN\\train_test17_18.xlsx')
w_sheet = excel_file.Activesheet

WIDTH, HEIGHT = 28, 28

train, test = 100000, 137050

label = []


for i in range(2,99999):
    if(w_sheet.Cells(i,1).Value==None):
        break

    st1 = w_sheet.Cells(i,3).Value # 회사코드 받기
    st2 = w_sheet.Cells(i, 10).Value  # 사업자등록번호 받기
    st3 = w_sheet.Cells(i, 21).Value  # 카테고리 받기
    st1 = str(st1)
    st2 = str(st2) #스트링 형식으로
    st3 = str(st3)

    de = w_sheet.Cells(i,16).Value # 라벨 받아오기
    de = int(de.split()[0])
    #print(st1+st2+st3)
    #print(de)

    label.append(de)

    MODEL_DIR = './img_17_18_2'
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    #if de != 146 and de != 827 :  continue
    encoded_image_path = text_to_image.encode(st1+st2+st3, "C:\\Users\\ialab\\PycharmProjects\\insu_CNN\\img_17_18_2\\a_%d_%d.png" %(test,de)) #이미지로 인코딩
    img = Image.open("C:\\Users\\ialab\\PycharmProjects\\insu_CNN\\img_17_18_2\\a_%d_%d.png" %(test,de)) # 이미지 사이즈 조절을 위해 이미지 다시 받아오기
    re_img = img.resize((28,28)) # 이미지 크기 설정
    re_img.save("C:\\Users\\ialab\\PycharmProjects\\insu_CNN\\img_17_18_2\\a_%d_%d.png" %(test,de))# 변경된 이미지 다시 저장
    test = test + 1

excel_file.Save()
excel.Quit()
