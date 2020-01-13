#import load_cnn
#import load_rf
from excel_join import ExcelJoin
#from excel_join_part2 import ExChJson
from load_cnn import CNN
from load_rf import RF
from RF_train import RF_train
from toimg import img
from filter_cnn import FILTER_CNN
from filter_rf import FILTER_RF
from Tax import tax
import pandas as pd
import os
import sys


from excel_file_creator import JsonToExcel
from file_seg import FileSeg
from file_seg_2 import FileSeg2

from json_convert import Convert
from ApplyUTF import Utf8Apply
import requests, json


PATH = "./process/"
excel_PATH = './pre/'
img_PATH = "./img/"
path_json = './json/'
RES_save_path = '/T-friend_data/RES/dev/'
RES_save_train_path = '/T-friend_data/TRAIN/res/'
path_2 = "./"


def pre_data(commend,proc, name = ''):

    ret = 0

    if proc == 0 :
        path = '/T-friend_data/REQ/dev/'  # 전달받은 자료 경로
        if commend != 'test':
            path = '/T-friend_data/TRAIN/dev/'

        #file_list = os.listdir(path)
        #last_len  = len(file_list)-1
        #print('file : ', file_list[last_len])

        #name = file_list[last_len]

        j = JsonToExcel(commend, path,path_2, name, proc)
        j.ToExecl()
        f = FileSeg()
        f.file_seg(commend, PATH, path_2)
        f2 = FileSeg2()
        ret = f2.file_seg2(PATH)

        return ret, name

    if proc == 1 :
        j2 = JsonToExcel(RES_save_path,path_2, name, proc)
        j2.ToExecl()

    return


def model_part(filename, comend, re_file='train'):
    ### 파일 불러오기(pandas)


    #filename = '12_file.xlsx'  ## 영수증
    #filename2 = 'e_bill_2019.xlsx'  ## 계산서
    # filename3 = 'total_17_18.xlsx'
    #filename3 = 'e_train_data.xlsx'
    # comend = 'test'
    filter_name_list = re_file.split('.')
    filter_name = filter_name_list[0]

    if filename == '12':
        filename = '12_file.json'
        filter_name = filter_name +'_12'
    elif filename == '34':
        filename = '34_file.json'
        filter_name = filter_name + '_34'
    else :
        filename = 'etc_file.json'
        filter_name = filter_name + '_etc'

    img_full_name = PATH + filename
    #df = pd.read_excel(img_full_name, encoding='utf-8', index_col=0)
    df = pd.read_json(img_full_name, orient='records')

    ### fearture에 품명이 있는지 검사(있으면 계산서, 없으면 영수증 취급)
    filename_spl = filename.split('_')
    # col_list = df.columns.tolist()
    # T = '영수증'

    T = '영수증'
    if filename_spl[0] == '12': T = '계산서'
    if filename_spl[0] == '34': T = '영수증'
    if filename_spl[0] == 'etc': T = '기타'

    # for i in range(len(col_list)):
    #    if col_list[i] == '품명' : T = '계산서'

    print('12_file.json 확인 : ', df.head())
    ### 이미지화
    a = img(T, df, comend, excel_PATH, img_PATH)
    a.pre_img()

    ### 해당 종류의 CNN 불러와서 C37 예측
    b = CNN(img_PATH, T, excel_PATH, img_full_name, comend)
    b.main_cnn()

    if comend == 'train':
       b = CNN(img_PATH=img_PATH, T=T, excel_PATH=excel_PATH, img_full_name=img_full_name, comend='test')
       b.main_cnn()

    ### CNN정확도 낮은 것 따로 빼놓음
    filter_name = filter_name + '_c'
    print('cnn_filter : '+filter_name)
    c = FILTER_CNN(T, excel_PATH, filter_name)
    c.main_f_cnn()

    d1 = RF_train(T, excel_PATH)
    d2 = RF(T, excel_PATH)

    if comend == 'train':
        ### 해당 종류의 RF 불러와서 결과 예측
        d1.main_RF_train()
    else:
        ### 해당 종류의 RF 불러와서 결과 예측
        d2.main_RF()

        ### RF정화도가 낮은 것 따로 빼놓음
        filter_name = filter_name + '_R'
        print('RF_filter : '+filter_name)
        e = FILTER_RF(T, excel_PATH, filter_name)
        e.main_f_rf()

    ### 공제/불공제 예측 ###
    #tax_filename = 'cash_train.xlsx'
 
    if filename == '34_file.json':
        tax_filename = 'cash_train.json'
    elif filename == 'etc_file.json':
        tax_filename = 'etc.json'
    if comend != 'train' and filename != '12_file.json':
        f = tax(excel_PATH, tax_filename)
        f.tax_predict()
    
    '''
    if comend != 'train' and filename == '34_file.xlsx':
        tax_filename = 'cash_train.xlsx'
        f = tax(excel_PATH, tax_filename)
        f.tax_predict()
    '''

def toRES(commend, name, etc):

    r_path = RES_save_path + name
    if commend != 'test':
        r_path = RES_save_train_path + name
        f = open(r_path, 'w')
        f.close()

    c = Convert(path_json, excel_PATH, PATH)
    c.convert(etc)
    u = Utf8Apply(r_path, path_json)
    u.utf_app()

def signal_in(tr_file, sysPath):
    url = sysPath

    payload = {"signal": tr_file}
    headers = {
        'Content-Type': "application/x-www-form-urlencoded",
        'User-Agent': "PostmanRuntime/7.17.1",
        'Accept': "*/*",
        'Cache-Control': "no-cache",
        'Postman-Token': "3d81ff59-a7e8-4a4e-8eee-109a54966340,e56720a2-44f5-4e21-875f-a150a93bca26",
        'Host': "dev-tms.3friend.co.kr",
        'Accept-Encoding': "gzip, deflate",
        'Content-Length': "39",
        'Cookie': "connect.sid=s%3A58t88RrAn1R8vBLeCyNhbVLrP9qyUotL.2nglp4FoBS2ZYQjU%2BJR8WBDs20JzvtUvhsz9yi%2F31G8",
        'Connection': "keep-alive",
        'cache-control': "no-cache"
        }

    response = requests.request("POST", url, data=payload, headers=headers)

def main(systemName, string, comment = 'test'):
   #re_file = 'A_20190925173904.REQ'
   #tr_file = 'A_test.RES'

   #comment = 'test' # 'test' or 'train'
   #train_data = '12' # '12' or '34' or 'etc'
   if systemName == "dev" :
      sysPath = 'https://dev-tms.3friend.co.kr/api/notify'
   elif systemName == "prod" :
      sysPath = 'https://tms.3friend.co.kr/api/notify'

   if comment == 'test':
       ret, last_file = pre_data(comment,0,string)

       filter_name_list = last_file.split('.')
       tr_file = filter_name_list[0]+'.RES'

       model_part('12', comment, last_file)
       model_part('34', comment, last_file)
       if ret == 1 :
           model_part('etc', comment, last_file)

       toRES(comment, tr_file, ret)

       signal_in(tr_file, sysPath)

       pre_data(1, tr_file)
   # api 
   else: # train 시
       EJ = ExcelJoin(string)
       EJ.ex_join()
       #ExChJson()
 
       ret, last_file = pre_data(comment,0,string)

       model_part('12', comment)
       model_part('34', comment)

       filter_name_list = last_file.split('.')
       tr_file = filter_name_list[0]+'.RES'

       toRES(comment, tr_file, ret)

       signal_in(tr_file, sysPath)

    
if __name__ == "__main__" :

#   try :
#      main(sys.argv[1], sys.argv[2], sys.argv[3])
#   except IndexError :
#      main(sys.argv[1], sys.argv[2])
   main(sys.argv[1], sys.argv[2], sys.argv[3])

