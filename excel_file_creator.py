import json
import pandas as pd



class JsonToExcel(object):
    def __init__(self, path, path_2, name, proc):
        self.path = path
        self.name = name
        self.proc = proc
        self.path_2 = path_2

    def ToExecl(self):

        file = self.path + self.name
        # data = open(file)
        # contents = json.load(data)

        # with open(file) as open_json:
        with open(file, encoding='UTF8') as open_json:
            contents = json.load(open_json)

            data1 = contents
            pd_data = pd.DataFrame(data1)
            print(pd_data)

            col_list = pd_data.columns.tolist()
            print(col_list)

            a = 0
            b = 0
            for i in range(len(col_list)):
                if col_list[i] == 'TP_BIZ':
                    a = 1
                elif col_list[i] == 'TP_BIZ_C':
                    b = 1

            if a ==0 :
                print('일반')
                #pd_data['TP_BIZ'] = '1'
            if b ==0 :
                print('일반')
                pd_data['TP_BIZ_C'] = ''

            PATH = '/home/cent/Documents/github/T-friend/'

            if self.proc == 0:
                pd_data['CD_ACCOUNT'] = ''
                pd_data['CD_ACCOUNT_R'] = ''
                pd_data['CD_DEDU'] = ''
                pd_data.to_excel(PATH + 'test_file.xlsx', 'w', encoding='utf-8')
            elif self.proc == 1:
                print("final : ",pd_data.head())
                pd_data.to_excel(PATH + 'result_file.xlsx', 'w', encoding='utf-8')


            # print(pd_data)
