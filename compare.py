################### 37종의 계정과목에 해당하는 내용만 남김(추가로 중복내용도 제거)########################

import pandas

list = ['141', '146', '150', '166', '172', '192', '194', '196', '198', '202', '209', '211', '254', '260', '267', '387',
        '401', '615', '715', '727', '809',
        '812', '813', '814', '817', '818', '819', '820', '821', '822', '823', '824', '825', '826', '827', '828', '831']
'''
raw_DATA = 'e_bill_2019.xlsx'
excel_PATH = "C:\\Users\\ialab\\Desktop\\T_Friend_data\\T-Friend\\3friend_raw_data\\"
excel_PATH2 = 'C:\\Users\\ialab\\Desktop\\T_Friend_data\\업종코드_매칭\\'
img_PATH = "C:\\Users\\ialab\\Desktop\\T_Friend_data\\img\\img_no2_test\\"
predict_PATH = "C:\\Git\\T-Friend\\predict_excel\\"
predict_FILE = 'tarin_industry_code.xlsx'
'''

def comp(comend, excel_PATH,T,df_1, loca, e_name, cate, cate2=0):
    list_mismatch = []
    if comend == 'train':
        if (T == '계산서'):
            # df = df_1.loc[:, ['품명', cate, cate2, loca]]
            #df = df_1.loc[:, [cate, cate2,'ITEM','CD_INDUSTRY','TP_BIZ_C',loca]].astype('str')
            df = df_1.loc[:, [cate, cate2,'NM_ITEM','CD_INDUSTRY',loca]].astype('str')
        else:
            df = df_1.loc[:, [cate, cate2,'TP_BIZ_C',loca]].astype('str')
    else :
        if (T == '계산서'):
            # df = df_1.loc[:, ['품명', cate, cate2, loca]]
            #df = df_1.loc[:, [cate, cate2,'ITEM','CD_INDUSTRY','TP_BIZ_C']].astype('str')
            df = df_1.loc[:, [cate, cate2,'NM_ITEM','CD_INDUSTRY']].astype('str')
        else:
            df = df_1.loc[:, [cate, cate2,'TP_BIZ_C']].astype('str')

    if comend == 'train':
        for i in range(len(df)):

            print("전처리 : %d/%d" % (i, len(df)))
            for j in range(len(list)):
                try:
                    #### 계정과목이 숫자와 문자열이 경우#####
                    ind = df.loc[i, [loca]].str.split(' ', n=2, expand=True)
                    if ind[0].item() == list[j]:  ###37종 계정과목에 맞는것만 골라냄
                        break
                    if j == (len(list) - 1):
                        list_mismatch.append(i)
                    #####################################

                except AttributeError:
                    #### 계정과목이 숫자만 있는 경우##################
                    ind = df.loc[i, [loca]].astype('str')
                    # print(ind)
                    if ind.item() == list[j]:
                        break
                    if j == (len(list) - 1):
                        list_mismatch.append(i)
                    #######################################


        print(len(list_mismatch))
        df = df.drop(list_mismatch)

    df = df.reset_index()
    df = df.drop(['index'], axis=1)

    print(df.head())
    if comend == 'test':
        #df.to_excel(excel_PATH + e_name)
        df.to_json(excel_PATH + e_name, orient='records', double_precision=15, default_handler=callable,force_ascii=False)

    return df
