import pandas as pd
import numpy as np

d18_train = 'train_test18.xlsx'
d17_train = 'train_test17.xlsx'
raw_DATA = '3friend_raw_data.xlsx'
code_DATA = '업종코드.xlsx'
excel_PATH = "C:\\Git\\T-Friend\\3friend_raw_data\\"
img_PATH = "C:\\Git\\T-Friend\\img_data_predict\\"
predict_PATH = "C:\\Git\\T-Friend\\predict_excel\\"
predict_FILE = 'credit_cash_predict.xlsx'


raw_data = pd.read_excel(excel_PATH + code_DATA, encoding='utf-8')
data_17 = pd.read_excel(excel_PATH + d17_train, encoding='utf-8')
data_18 = pd.read_excel(excel_PATH + d18_train, encoding='utf-8')

buyer_17 = data_17['사업자번호'].str.split('-', n=2, expand=True)
buyer_17['사업자번호'] = buyer_17[0].str.cat(buyer_17[1])
buyer_17['사업자번호'] = buyer_17['사업자번호'].str.cat(buyer_17[2]).copy()

del(data_17['사업자번호'])

data_17['사업자번호'] = buyer_17['사업자번호']

print(data_17.loc[1,:])