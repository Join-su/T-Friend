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
            name = '사업자등록번호'
            train = pd.read_excel(self.excel_PATH + raw_DATA, encoding='utf-8', sheet_name='Sheet1')
            X = train.loc[:, [name,'업종코드', '사업자번호', 'cc']].copy()

            '''
            X['회사구분'] = LabelEncoder().fit_transform(X['회사구분'])  # 종류별로 column 나// LabelEncoder() - 0~7로 라벨을 바꿔줌(one-hot 전 단계), fit_transform() - 변경시킬 데이터를 받아주는 역할
            X3 = pd.DataFrame(LabelBinarizer().fit_transform(X['회사구분']), columns=['회사구분0', '회사구분1', '회사구분2', '회사구분3']
                              , index=X.index)  # X.index - X(데이터 정보가 들어있는)의 인덱스를 기준으로 매칭시킬것
            X = pd.concat([X, X3], axis=1)  # 결합  axis=1 : column
            del (X['회사구분'])
            '''

        else:
            raw_DATA = 'cash_train.xlsx'
            save_file = 'cash_train_model.sav'
            name = '회사등록번호'
            train = pd.read_excel(self.excel_PATH + raw_DATA, encoding='utf-8', sheet_name='Sheet1')
            X = train.loc[:, [name, '사업자번호', 'cc']].copy()


        try :###실전에서는 계정과목 존재 X
            X_label = train.loc[:, ['계정과목']].copy()
            X_label.dropna(inplace=True)
            index = X_label['계정과목'].str.split(' ', n=1, expand=True)
            lists = list(index)
            Y = index[0].copy()
        except AttributeError:
            Y = train.loc[:, ['계정과목']].copy()


        try :
            num = X[name].str.split('-', n=2, expand=True)
            num['거래처번호'] = num[0].str.cat(num[1])
            num['거래처번호'] = num['거래처번호'].str.cat(num[2])

            X = pd.concat([X, num['거래처번호']], axis=1)
            #X = X.rename(columns={'사업자등록번호':'거래처번호'})
        except AttributeError :
            X['거래처번호'] = X[name]


        del(X[name])


        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)


        loaded_model = pickle.load(open('C:\\Users\\ialab\\PycharmProjects\\Total\\RF_save\\' + save_file, 'rb'))


        print(X.head())
        predict = loaded_model.predict(X)
        '''
        print('predict : ', predict)
        df = pd.DataFrame(predict)
        df.columns = ['예측값']
        print(df)
        #df['정답'] = Y
        train = pd.concat([train, df['예측값']], axis=1)
        '''
        train['예측값'] = predict.astype('int')
        train['예측정도'] = 0

        pre = loaded_model.predict_proba(X)
        for i in range(len(pre)):
            pre_val = np.argmax(pre[i])
            train.loc[i, ['예측정도']] = pre[i][pre_val]

        train.to_excel(self.excel_PATH + raw_DATA)

        # print('pre_len : ', len(pre[0]))
        # print(pre[1])
        print(f'Out-of-bag score estimate: {loaded_model.oob_score_:.3}')
        print(classification_report(y_train.astype('int'), loaded_model.predict(X_train).astype('int')))

        '''
        result = loaded_model.score(X, Y) # acc 출력
        print(result)
        fout = open('C:\\Users\\ialab\\Desktop\\T_Friend_data\\업종코드_매칭\\cash_test_result.txt', 'a')
        print('tr18, te 18 : %f' %result, file=fout)
        fout.close()
        '''
