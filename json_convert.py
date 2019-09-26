import pandas as pd

json_name = 'A_20190925173904.RES'
'''
file_12 = '12_file.xlsx'
data = pd.read_excel(file_12, index_col=0)
data.to_json('12.json', orient='records', double_precision=15, default_handler=callable, force_ascii=False)

file_34 = '34_file.xlsx'
data1 = pd.read_excel(file_34, index_col=0)
data1.to_json('34.json', orient='records', double_precision=15, default_handler=callable, force_ascii=False)

file_extra = 'extra.xlsx'
data2 = pd.read_excel(file_extra, index_col=0)
data2.to_json('extra.json', orient='records', double_precision=15, default_handler=callable, force_ascii=False)
'''

file_extra = 'test_file.xlsx'
data2 = pd.read_excel(file_extra, index_col=0)
data2.to_json('test_file.json', orient='records', double_precision=15, default_handler=callable, force_ascii=False)


#print(data)