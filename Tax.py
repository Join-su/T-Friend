import pandas as pd

#PATH = 'C:\\Users\\ialab\\Desktop\\T-Friend\\pre\\'


class tax(object):
    def __init__(self, excel_PATH ,filename):
        self.PATH = excel_PATH
        self.filename = filename

    def tax_predict(self) :

        # filename = 'e_bill_2019_uniq.xlsx'
        # filename = 'cash_train.xlsx'

        df = pd.read_excel(self.PATH + self.filename, encoding='utf-8', index_col=0)

        txt_812 = ['항공', '여객기', '버스', '운송', '택시', '이비카드', '한국스마트']
        txt_252 = ['금융결제원']
        txt_826 = ['도서', '문고', '교육', '영어', '수학', '학원']

        count = 0

        for i in range(len(df)):
            if df.loc[i, ['TP_BIZ_C']].item() == 1 or df.loc[i, ['TP_BIZ_C']].item() == 3:
                #print('1, 3')
                if df.loc[i, ['CD_ACCOUNT']].item() == 813:
                    df.loc[i, ['CD_DEDU']] = 1
                    count += 1
                elif df.loc[i, ['CD_ACCOUNT']].item() == 812:
                    string = df.loc[i, ['NM_COMP_C']].item()
                    for j in txt_812:
                        if any(j in s for s in string):
                            df.loc[i, ['CD_DEDU']] = 1
                            count += 1
                elif df.loc[i, ['CD_ACCOUNT']].item() == 252:
                    string = df.loc[i, ['NM_COMP_C']].item()
                    for j in txt_252:
                        if any(j in s for s in string):
                            df.loc[i, ['CD_DEDU']] = 1
                            count += 1
                elif df.loc[i, ['CD_ACCOUNT']].item() == 826:
                    string = df.loc[i, ['NM_COMP_C']].item()
                    for j in txt_826:
                        if any(j in s for s in string):
                            df.loc[i, ['CD_DEDU']] = 1
                            count += 1
                elif df.loc[i, ['CD_ACCOUNT']].item() == 250:
                    df.loc[i, ['CD_DEDU']] = ''
                else:
                    df.loc[i, ['CD_DEDU']] = 0
            elif df.loc[i, ['TP_BIZ_C']].item() == 2 or df.loc[i, ['TP_BIZ_C']].item() == 4:
                if df.loc[i, ['CD_ACCOUNT']].item() == 146:
                    df.loc[i, ['CD_DEDU']] = 0
                elif df.loc[i, ['CD_ACCOUNT']].item() == 250:
                    df.loc[i, ['CD_DEDU']] = ''
                else:
                    df.loc[i, ['CD_DEDU']] = 1
            else :
                if df.loc[i, ['CD_ACCOUNT']].item() == 812:
                    string = df.loc[i, ['NM_COMP_C']].item()
                    for j in txt_812:
                        if any(j in s for s in string):
                            df.loc[i, ['CD_DEDU']] = 1
                            count += 1
                elif df.loc[i, ['CD_ACCOUNT']].item() == 252:
                    string = df.loc[i, ['NM_COMP_C']].item()
                    for j in txt_252:
                        if any(j in s for s in string):
                            df.loc[i, ['CD_DEDU']] = 1
                            count += 1
                elif df.loc[i, ['CD_ACCOUNT']].item() == 826:
                    string = df.loc[i, ['NM_COMP_C']].item()
                    for j in txt_826:
                        if any(j in s for s in string):
                            df.loc[i, ['CD_DEDU']] = 1
                            count += 1


        print(df.head())
        print(count)
        df.to_excel(self.PATH + self.filename, 'w', encoding='utf-8')
