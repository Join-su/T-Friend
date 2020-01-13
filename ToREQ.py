import json
#import pandas as pd

path = '/home/cent/Documents/github/T-friend/train_data/'
file = path + 'RES.json'
#file = "C:\\Users\\ialab\\Desktop\\T-Friend\\json\\RES.json"

#DATA_NAME = 'A_test.RES'

#data = pd.read_json(file, orient='records')
#print(data.head())

with open(file) as json_file:
    data = json.load(json_file)
    #print(data)
    with open(path + 'test.REQ', 'w', encoding='UTF8') as write_file:
        write_file.write(json.dumps(data, ensure_ascii=False))
