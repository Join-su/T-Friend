import pandas as pd

class FILTER_RF(object):
    def __init__(self,T, excel_PATH):
        self.T = T
        self.excel_PATH = excel_PATH

    def main_f_rf(self):
        if self.T == '계산서':
            raw_DATA = 'e_bill_2019_uniq.xlsx'
            e_name = '계산서_filter_rf.xlsx'
        else:
            raw_DATA = 'cash_train.xlsx'
            e_name = '영수증_filter_rf.xlsx'

        df = pd.read_excel(self.excel_PATH + raw_DATA, encoding='utf-8', sheet_name='Sheet1', index_col=0)

        list = []
        count = 0
        print('Selecting less than 80% accuracy...(RF)')
        for i in range(len(df)):
            if float(df.loc[i,['예측정도']].item())<0.8:
                list.append(i)
                count +=1
        print('count : ',count)

        df_1 = df.iloc[list,:]

        df_1.to_excel('C:\\Users\\ialab\\PycharmProjects\\Total\\filter_rf\\' + e_name)



