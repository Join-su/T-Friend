import pandas as pd

path = '/home/cent/Documents/github/T-friend/train_data/'
filename = 'e_train_data_2.xlsx'

data = pd.read_excel(path + filename, index_col=0)

data.to_json(path + 'RES.json', orient='records', double_precision=15, default_handler=callable,force_ascii=False)



