###### json file을 RES로 (UTF8로 읽히게) 만듦####

import json



class Utf8Apply(object):
    def __init__(self, DATA_NAME):
        self.DATA_NAME = DATA_NAME

    def utf_app(self):
        file = "C:\\Users\\ialab\\Desktop\\T-Friend\\json\\RES.json"

        #DATA_NAME = 'A_test.RES'

        with open(file) as json_file:
            data = json.load(json_file)
            print(data)
            with open(self.DATA_NAME, 'w', encoding='UTF8') as write_file:
                write_file.write(json.dumps(data, ensure_ascii=False))
