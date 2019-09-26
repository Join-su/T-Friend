import json

data = 'A_20190925173904.RES'


with open(data)as json_file:
    data2 = json.load(json_file)
    with open('test.RES', 'w', encoding='UTF8') as file:
        file.write(json.dumps(data2, ensure_ascii=False))


print(data2)