import json
import pandas as pd

file = 'A_20190919105444.REQ'

data = open(file, encoding='UTF8')
contents = json.load(data)

with open("test.json", 'w') as save_json:
    json.dump(contents, save_json)

file = 'test.json'

with open(file) as json_file:
    data = json.load(json_file)
    pd_data = pd.DataFrame(data)
    pd_data.to_excel('test_file.xlsx', 'w', encoding='utf-8')
    print(pd_data)
