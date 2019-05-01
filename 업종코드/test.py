import pandas as pd
import numpy as np

d18_train = 'train_test18.xlsx'
d17_train = 'train_test17.xlsx'
raw_DATA = '3friend_raw_data.xlsx'
code_DATA = '업종코드.xlsx'
excel_PATH = "C:\\Git\\T-Friend\\3friend_raw_data\\"
img_PATH = "C:\\Git\\T-Friend\\img_data_predict\\"
predict_PATH = "C:\\Git\\T-Friend\\predict_excel\\"
test_PATH = 'C:\\Git\\T-Friend\\test\\'
test_FILE= 'test_18.xlsx'
predict_FILE = 'credit_cash_predict.xlsx'


raw_data = pd.read_excel(excel_PATH + code_DATA, sheet_name='추출', encoding='utf-8')
data_17 = pd.read_excel(excel_PATH + d17_train, encoding='utf-8')
data_18 = pd.read_excel(excel_PATH + d18_train, encoding='utf-8')

# 17년도 데이터 사업자번호 추출
buyer_17 = data_17['사업자번호'].str.split('-', n=2, expand=True)
buyer_17['사업자번호'] = buyer_17[0].str.cat(buyer_17[1])
buyer_17['사업자번호'] = buyer_17['사업자번호'].str.cat(buyer_17[2]).copy()
del(buyer_17[0])
del(buyer_17[1])
del(buyer_17[2])
buyer_17 = buyer_17.astype('str')

# 17년도 사업자번호 리스트(중복제거)
num_17 = buyer_17['사업자번호']
num_17 = np.unique(num_17)
num_17 = pd.DataFrame(num_17)
num_17.columns = ['사업자번호']

# 17년도 데이터 정제
del(data_17['Unnamed: 0'])
del(data_17['사업자번호'])
data_17['사업자번호'] = buyer_17['사업자번호'].astype('str')
data_17['업종코드'] = 0

# 18년도 데이터 사업자번호 추출
buyer_18 = data_18['사업자번호'].str.split('-', n=2, expand=True)
buyer_18['사업자번호'] = buyer_18[0].str.cat(buyer_18[1])
buyer_18['사업자번호'] = buyer_18['사업자번호'].str.cat(buyer_18[2]).copy()
del(buyer_18[0])
del(buyer_18[1])
del(buyer_18[2])
buyer_18 = buyer_18.astype('str')

# 18년도 데이터 사업자번호 추출
del(data_18['Unnamed: 0'])
del(data_18['사업자번호'])
data_18['사업자번호'] = buyer_18['사업자번호'].astype('str')
data_18['업종코드'] = 0

# 18년도 사업자번호 리스트(중복제거)
num_18 = buyer_18['사업자번호']
num_18 = np.unique(num_18)
num_18 = pd.DataFrame(num_18)
num_18.columns = ['사업자번호']

# 업종코드 보유한 사업자번호 리스트
code_data = raw_data
number = np.unique(code_data)
number = pd.DataFrame(number)
number.columns = ['사업자번호']
number = number.astype('str')
code_data = raw_data.astype('str')


count = 0
for i in range(len(data_18)):
    print('%d / %d' % (i, len(data_18)))
    for j in range(len(code_data)):
        if data_18.loc[i, ['사업자번호']].item() == code_data.loc[j, ['사업자번호']].item():
            count = count + 1
            print(count)
            data_18.loc[i, ['업종코드']] = code_data.loc[j, ['업종코드']]



data_18.to_excel(test_PATH + test_FILE)
print(count)
