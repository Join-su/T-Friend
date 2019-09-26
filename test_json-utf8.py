###### json file을 RES로 (UTF8로 읽히게) 만듦####

import json

file = "C:\\Users\\ialab\\Desktop\\T-Friend\\test_file.json"

DATA_NAME = 'A_20190925173904.RES'


with open(file) as json_file:
    data = json.load(json_file)
    print(data)
    with open(DATA_NAME, 'w', encoding='UTF8') as write_file:
        write_file.write(json.dumps(data, ensure_ascii=False))
