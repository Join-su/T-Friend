#import load_cnn
#import load_rf
from load_cnn import CNN
from load_rf import RF
from RF_train import RF_train
from toimg import img
from filter_cnn import FILTER_CNN
from filter_rf import FILTER_RF
from Tax import tax
import pandas as pd
import os


from excel_file_creator import JsonToExcel
from file_seg import FileSeg
from file_seg_2 import FileSeg2

from json_convert import Convert
from ApplyUTF import Utf8Apply



PATH = "C:\\Users\\ialab\\Desktop\\T-Friend\\process\\"
excel_PATH = 'C:\\Users\\ialab\\Desktop\\T-Friend\\pre\\'
img_PATH = "C:\\Users\\ialab\\Desktop\\T-Friend\\img\\"
path_json = 'C:\\Users\\ialab\\Desktop\\T-Friend\\json\\'
RES_save_path = 'C:\\Users\\ialab\\Desktop\\T-Friend\\RES\\'


def pre_data(proc, name = ''):

    ret = 0

    if proc == 0 :
        path = 'C:\\Users\\ialab\\Desktop\\T-Friend\\REQ\\'  # 전달받은 자료 경로

        file_list = os.listdir(path)
        last_len  = len(file_list)-1
        print('file : ', file_list[last_len])

        name = file_list[last_len]

        j = JsonToExcel(path, name, proc)
        j.ToExecl()
        f = FileSeg()
        f.file_seg(PATH)
        f2 = FileSeg2()
        ret = f2.file_seg2(PATH)

        return ret, file_list[last_len]

    if proc == 1 :
        # 최종본 excel로 저장하는 경로
        j = JsonToExcel(RES_save_path, name, proc)
        j.ToExecl()

    return


def model_part(filename, comend, re_file):
    ### 파일 불러오기(pandas)


    #filename = '12_file.xlsx'  ## 영수증
    #filename2 = 'e_bill_2019.xlsx'  ## 계산서
    # filename3 = 'total_17_18.xlsx'
    #filename3 = 'e_train_data.xlsx'
    # comend = 'test'
    filter_name_list = re_file.split('.')
    filter_name = filter_name_list[0]

    if filename == '12':
        filename = '12_file.xlsx'
        filter_name = filter_name +'_12'
    elif filename == '34':
        filename = '34_file.xlsx'
        filter_name = filter_name + '_34'
    else :
        filename = 'etc_file.xlsx'
        filter_name = filter_name + '_etc'

    img_full_name = PATH + filename
    df = pd.read_excel(img_full_name, encoding='utf-8', index_col=0)

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

    print(df.head())
    ### 이미지화
    a = img(T, df, comend, excel_PATH, img_PATH)
    a.pre_img()

    ### 해당 종류의 CNN 불러와서 C37 예측
    b = CNN(img_PATH, T, excel_PATH, img_full_name, comend)
    b.main_cnn()

    ### CNN정확도 낮은 것 따로 빼놓음
    filter_name = filter_name + '_c'
    print('cnn_filter : '+filter_name)
    c = FILTER_CNN(T, excel_PATH, filter_name)
    c.main_f_cnn()

    if comend == 'train':
        ### 해당 종류의 RF 불러와서 결과 예측
        d1 = RF_train(T, excel_PATH)
        d1.main_RF_train()
    else:
        ### 해당 종류의 RF 불러와서 결과 예측
        d2 = RF(T, excel_PATH)
        d2.main_RF()

    ### RF정화도가 낮은 것 따로 빼놓음
    filter_name = filter_name + '_R'
    print('RF_filter : '+filter_name)
    e = FILTER_RF(T, excel_PATH, filter_name)
    e.main_f_rf()

    ### 공제/불공제 예측 ###
    if filename == '12_file.xlsx':
        tax_filename = 'e_bill_2019_uniq.xlsx'
    elif filename == '34_file.xlsx':
        tax_filename = 'cash_train.xlsx'
    elif filename == 'etc_file.xlsx':
        tax_filename = 'etc.xlsx'
    f = tax(excel_PATH, tax_filename)
    f.tax_predict()


def toRES(name, etc):

    r_path = RES_save_path + name

    c = Convert(path_json, excel_PATH, PATH)
    c.convert(etc)
    u = Utf8Apply(r_path, path_json)
    u.utf_app()



#re_file = 'A_20190925173904.REQ'
#tr_file = 'A_test.RES'

ret, last_file = pre_data(0)

filter_name_list = last_file.split('.')
tr_file = filter_name_list[0]+'.RES'

model_part('12', 'test', last_file)
model_part('34', 'test', last_file)
if ret == 1 :
    model_part('etc', 'test', last_file)

toRES(tr_file, ret)

pre_data(1, tr_file)

