import pandas as pd
import sys


def main(path, filename):

    df = pd.read_json(path + filename, orient='records')
    df.to_excel(path + 'View.xlsx', 'w', encoding='utf-8')

if __name__=="__main__" :
    
    main(sys.argv[1], sys.argv[2])
