import json
import pandas as pd

file = 'C:\\Users\\ialab\\Desktop\\T-Friend\\A_20190925173904.RES'

#data = open(file)
#contents = json.load(data)

#with open(file) as open_json:
with open(file, encoding='UTF8') as open_json:
    contents = json.load(open_json)

    data1 = contents
    pd_data = pd.DataFrame(data1)
    print(pd_data)

    col_list = pd_data.columns.tolist()
    print(col_list)
    for i in range(len(col_list)):
        if col_list[i] != 'TP_BIZ' :
            print('TP_BIZ')
            if i == len(col_list) - 1:
                print('일반')
                pd_data['TP_BIZ'] = '1'
    #pd_data['CD_ACCOUNT'] = 0
    #pd_data['CD_DEDU'] = 0

    pd_data.to_excel('test_file.xlsx', 'w', encoding='utf-8')
    #print(pd_data)