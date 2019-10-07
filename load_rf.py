import pandas as pd
import pickle
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


PATH = 'C:\\Users\\aiia\\.atom\\python\\data_keyword\\'
excel_PATH = 'C:\\Users\\aiia\\.atom\\python\\data_keyword\\'
test_PATH = 'C:\\Users\\ialab\\PycharmProjects\\dtree\\'
code_PATH = 'C:\\Users\\ialab\\Desktop\\T_Friend_data\\업종코드_매칭\\'
d18_train = 'total_17_18_comp.xlsx'


class RF(object):
    def __init__(self,T, excel_PATH):
        self.T = T
        self.excel_PATH = excel_PATH

    def main_RF(self):
        if self.T == '계산서':
            raw_DATA = 'e_bill_2019_uniq.xlsx'
            save_file = 'new_RF_model.sav'
            name = 'NO_BIZ_C'
            name2 = 'NO_BIZ'
            train = pd.read_excel(self.excel_PATH + raw_DATA, encoding='utf-8', sheet_name='Sheet1', index_col=0)
            #train.rename(columns={"NO_BIZ": "사업자등록번호"}, inplace=True)
            #train.rename(columns={"NO_BIZ_C_new": "NO_BIZ_C"}, inplace=True)
            #train.rename(columns={"CD_INDUSTRY": "업종코드"}, inplace=True)
            #train.rename(columns={"CD_ACCOUNT": "계정과목"}, inplace=True)
            X = train.loc[:, ['NO_BIZ_C', 'CD_INDUSTRY', 'TP_BIZ','NO_BIZ', 'cc']].copy()
            print(X.head())

            '''
            X['회사구분'] = LabelEncoder().fit_transform(X['회사구분'])  # 종류별로 column 나// LabelEncoder() - 0~7로 라벨을 바꿔줌(one-hot 전 단계), fit_transform() - 변경시킬 데이터를 받아주는 역할
            X3 = pd.DataFrame(LabelBinarizer().fit_transform(X['회사구분']), columns=['회사구분0', '회사구분1', '회사구분2', '회사구분3']
                              , index=X.index)  # X.index - X(데이터 정보가 들어있는)의 인덱스를 기준으로 매칭시킬것
            X = pd.concat([X, X3], axis=1)  # 결합  axis=1 : column
            del (X['회사구분'])
            '''

        elif self.T == '영수증':
            raw_DATA = 'cash_train.xlsx'
            save_file = 'cash_train_model.sav'
            name = 'NO_BIZ_C'
            name2 = 'NO_BIZ'
            train = pd.read_excel(self.excel_PATH + raw_DATA, encoding='utf-8', sheet_name='Sheet1', index_col=0)
            #train.rename(columns={"NO_BIZ": "회사등록번호"}, inplace=True)
            #train.rename(columns={"NO_BIZ_C_new": "NO_BIZ_C"}, inplace=True)
            #train.rename(columns={"CD_ACCOUNT": "계정과목"}, inplace=True)
            X = train.loc[:, ['NO_BIZ_C', 'TP_BIZ', 'NO_BIZ', 'cc']].copy()

        elif self.T == '기타':
            raw_DATA = 'etc.xlsx'
            save_file = 'cash_train_model.sav'
            name = 'NO_BIZ_C'
            name2 = 'NO_BIZ'
            train = pd.read_excel(self.excel_PATH + raw_DATA, encoding='utf-8', sheet_name='Sheet1', index_col=0)
            #train.rename(columns={"NO_BIZ": "회사등록번호"}, inplace=True)
            #train.rename(columns={"NO_BIZ_C_new": "NO_BIZ_C"}, inplace=True)
            #train.rename(columns={"CD_ACCOUNT": "계정과목"}, inplace=True)
            X = train.loc[:, ['NO_BIZ_C', 'TP_BIZ', 'NO_BIZ', 'cc']].copy()
        #print(X.head())


        try :###실전에서는 계정과목 존재 X
            X_label = train.loc[:, ['CD_ACCOUNT']].copy()
            X_label.dropna(inplace=True)
            index = X_label['CD_ACCOUNT'].str.split(' ', n=1, expand=True)
            lists = list(index)
            Y = index[0].copy()
        except AttributeError:
            Y = train.loc[:, ['CD_ACCOUNT']].copy()

        try :
            num = X[name].str.split('-', n=2, expand=True)
            num[name] = num[0].str.cat(num[1])
            num[name] = num[name].str.cat(num[2])
            del (X[name])
            X = pd.concat([X, num[name]], axis=1)
            #print('head',X.head())
            #X = X.rename(columns={'사업자등록번호':'거래처번호'})
        except AttributeError :
            X['NO_BIZ_C_new'] = X[name]
            del(X[name])

        #print(X.head())


        try :
            num = X[name2].str.split('-', n=2, expand=True)
            num[name2] = num[0].str.cat(num[1])
            num[name2] = num[name2].str.cat(num[2])
            del (X[name2])
            X = pd.concat([X, num[name2]], axis=1)
            #X = X.rename(columns={'사업자등록번호':'거래처번호'})
        except AttributeError :
            X['NO_BIZ_new'] = X[name2]
            del(X[name2])

        train.rename(columns={"NO_BIZ_new": "NO_BIZ"}, inplace=True)
        train.rename(columns={"NO_BIZ_C_new": "NO_BIZ_C"}, inplace=True)

        #print(X.head())


        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)


        loaded_model = pickle.load(open('/home/cent/Documents/github/T-friend/RF_save/' + save_file, 'rb'))


        print(X.head())
        predict = loaded_model.predict(X)

        print('predict : ', predict)
        #df = pd.DataFrame(predict)
        #df.columns = ['예측값']
        #print(df)
        #df['정답'] = Y
        #train = pd.concat([train, df['예측값']], axis=1)

        train['CD_ACCOUNT'] = predict.astype('int')
        train['예측정도'] = 0
        #train['recommend'] = 0

        pre = loaded_model.predict_proba(X)
        for i in range(len(pre)):
            pre_val = np.argmax(pre[i])
            train.loc[i, ['예측정도']] = pre[i][pre_val]

        train.to_excel(self.excel_PATH + raw_DATA)

        # print('pre_len : ', len(pre[0]))
        # print(pre[1])
        print(f'Out-of-bag score estimate: {loaded_model.oob_score_:.3}')
        #print(classification_report(y_train.astype('int'), loaded_model.predict(X_train).astype('int')))

        '''
        result = loaded_model.score(X, Y) # acc 출력
        print(result)
        fout = open('C:\\Users\\ialab\\Desktop\\T-Friend\\result\\cash_test_result.txt', 'a')
        print('tr18, te 18 : %f' %result, file=fout)
        fout.close()
        '''

