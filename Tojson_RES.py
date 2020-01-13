import pandas as pd
import sys


def main(path, filename):

    file_1 = path + filename
    with open(file_1, encoding='UTF8') as open_json:
            contents = pd.read_json(open_json, orient='records')
            df_1 = contents
    df_1.to_json(path + 'RES.json', orient='records', double_precision=15, default_handler=callable,force_ascii=False)
    df = pd.read_json(path + 'RES.json', orient='records')
    df.to_excel(path + 'View.xlsx', 'w', encoding='utf-8')

if __name__=="__main__" :

    main(sys.argv[1], sys.argv[2])

