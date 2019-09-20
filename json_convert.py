import pandas as pd

file = 'test_file2.xlsx'
data = pd.read_excel(file, index_col=0)
data.to_json('A_20190919105444.RES', orient='records')
print(data)
