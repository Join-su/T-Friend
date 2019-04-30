import pandas as pd
import re

PATH = 'c:\\3-f\\3friend_raw_data\\'
excel_PATH = 'c:\\3-f\\excel_data\\'
ori_data = '3friend_raw_data.xlsx'
extra_data = '3friend_extra_data.xlsx'
test_dtree_excel = 'test_dtree.xlsx'
train_data = 'data.xlsx'
keyword_data = 'keyword_table2.xlsx'
year17_data = '2017_data.xlsx'

DATA = pd.read_excel(excel_PATH + year17_data, encoding='utf-8')

test = DATA.loc[:, ['품명', '계정과목']].copy()
test = test.fillna('기타')
test = test.drop_duplicates()
test = test.reset_index()
del(test['index'])

key = pd.read_excel(excel_PATH + keyword_data)
categori_name = list(key)
for z in categori_name:

    for j in range(len(key)):
        keyword = str(key.loc[j, z])
        #print(keyword)
        if keyword == 'nan': break
        p = re.compile(keyword)
        for i in range(len(test)):
            b = str(test.loc[i, '품명'])
            a = p.search(b)
            if a != None:
                #print(a, i)
                test.loc[i, 'c'] = z

test.to_excel('keyword_2018_sheet3.xlsx')

