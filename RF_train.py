import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

'''
PATH = 'C:\\Users\\aiia\\.atom\\python\\data_keyword\\'
excel_PATH = 'C:\\Users\\aiia\\.atom\\python\\data_keyword\\'
test_PATH = 'C:\\Users\\aiia\\.atom\\python\\data_keyword\\result\\'
'''
code_PATH = 'C:\\Users\\ialab\\Desktop\\T-Friend\\RF_ACC\\'
excel_PATH = 'C:\\Users\\ialab\\Desktop\\T_Friend_data\\업종코드_매칭\\'
# excel_PATH = 'C:\\Users\\ialab\\Desktop\\T_Friend_data\\T-Friend\\3friend_raw_data\\'
d18_train = 'train_test18.xlsx'
d17_train = 'train_test17.xlsx'
d17_18_train = 'train_test17_18.xlsx'
year17_data = '2017_data.xlsx'
new_data = 'total_17_18_new.xlsx'


class RF_train(object):
    def __init__(self, T, excel_PATH, ):
        self.T = T
        self.excel_PATH = excel_PATH

    def main_RF_train(self):
        if self.T == '계산서':
            new_data = 'e_bill_2019_uniq.xlsx'
            filename = 'new_RF_model.sav'

            train = pd.read_excel(self.excel_PATH + new_data, encoding='utf-8')

            #X = train.loc[:, ['NO_BIZ_C', 'CD_INDUSTRY', 'TP_BIZ_C','NO_BIZ', 'cc']].copy()
            X = train.loc[:, ['NO_BIZ_C', 'CD_INDUSTRY','NO_BIZ', 'cc']].copy()
            X_label = train.loc[:, ['CD_ACCOUNT']].copy()
        else :
            new_data = 'cash_train.xlsx'
            filename = 'cash_train_model.sav'

            train = pd.read_excel(self.excel_PATH + new_data, encoding='utf-8')

            X = train.loc[:, ['NO_BIZ_C', 'TP_BIZ_C', 'NO_BIZ', 'cc']].copy()
            X_label = train.loc[:, ['CD_ACCOUNT']].copy()


        '''
        X_label.dropna(inplace=True)#빈곳 제거후 땡겨채운(inplace)
        index = X_label['계정과목'].str.split(' ', n=1, expand=True) #expand - 별도의 열로 확장, n - 몇번째 ''
        lists = list(index)
        Y = index[0].copy()
        '''
        # Y = X_label

        # X : dataset Y : label
        '''
        X['c'] = LabelEncoder().fit_transform(X['c'])#종류별로 column 나// LabelEncoder() - 0~7로 라벨을 바꿔줌(one-hot 전 단계), fit_transform() - 변경시킬 데이터를 받아주는 역할
        X2 = pd.DataFrame(LabelBinarizer().fit_transform(X['c']), columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7']
                          , index=X.index) #X.index - X(데이터 정보가 들어있는)의 인덱스를 기준으로 매칭시킬것
        X = pd.concat([X, X2], axis=1)#결합  axis=1 : column
        del(X['c'])
        '''

        '''
        X['회사구분'] = LabelEncoder().fit_transform(X['회사구분'])#종류별로 column 나// LabelEncoder() - 0~7로 라벨을 바꿔줌(one-hot 전 단계), fit_transform() - 변경시킬 데이터를 받아주는 역할
        X3 = pd.DataFrame(LabelBinarizer().fit_transform(X['회사구분']), columns=['회사구분0', '회사구분1', '회사구분2', '회사구분3']
                          , index=X.index) #X.index - X(데이터 정보가 들어있는)의 인덱스를 기준으로 매칭시킬것
        X = pd.concat([X, X3], axis=1)#결합  axis=1 : column
        del(X['회사구분'])
        '''

        try:  ###실전에서는 계정과목 존재 X
            X_label = train.loc[:, ['CD_ACCOUNT']].copy()
            X_label.dropna(inplace=True)
            index = X_label['CD_ACCOUNT'].str.split(' ', n=1, expand=True)
            lists = list(index)
            Y = index[0].copy()
        except AttributeError:
            Y = train.loc[:, ['CD_ACCOUNT']].astype('int').copy()

        try:
            num = X['NO_BIZ_C'].str.split('-', n=2, expand=True)
            num['NO_BIZ_C_new'] = num[0].str.cat(num[1])
            num['NO_BIZ_C_new'] = num['NO_BIZ_C_new'].str.cat(num[2])
            del (X['NO_BIZ_C'])
            X = pd.concat([X, num['NO_BIZ_C_new']], axis=1)
            # X = X.rename(columns={'사업자등록번호':'거래처번호'})
        except AttributeError:
            X['NO_BIZ_C_new'] = X['NO_BIZ_C']

            del(X['NO_BIZ_C'])

        try:
            num = X['NO_BIZ'].str.split('-', n=2, expand=True)
            num['NO_BIZ_new'] = num[0].str.cat(num[1])
            num['NO_BIZ_new'] = num['NO_BIZ_new'].str.cat(num[2])
            del (X['NO_BIZ'])
            X = pd.concat([X, num['NO_BIZ_new']], axis=1)
            # X = X.rename(columns={'사업자등록번호':'거래처번호'})
        except AttributeError:
            X['NO_BIZ_new'] = X['NO_BIZ']

            del(X['NO_BIZ'])

        X.rename(columns={"NO_BIZ_new": "NO_BIZ"}, inplace=True)
        X.rename(columns={"NO_BIZ_C_new": "NO_BIZ_C"}, inplace=True)

        print(X.head())

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
        # X_train, y_train = X,Y

        model = RandomForestClassifier(n_estimators=100, oob_score=True,
                                       random_state=123456)  # n_estimators=35<-라벨수, oob_score=True<-score출력, random_state=123456<-randum값 고정 시드
        model.fit(X_train, y_train)
        '''
        X_train.to_excel(test_PATH + 'train2017.xlsx')
        X_test.to_excel(test_PATH + 'test2017.xlsx')

        y_train.to_excel(test_PATH + 'train_label2017.xlsx')
        y_test.to_excel(test_PATH + 'test_label2017.xlsx')


        predict = model.predict(X_test)
        df = pd.DataFrame(predict)
        df.to_excel(test_PATH + "predict.xlsx")
        '''

        fout = open(code_PATH + 'result.txt', 'a')
        pickle.dump(model, open('/home/cent/Documents/github/save_file/' + filename, 'wb'))
        # model = DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_leaf=5).fit(X_train, y_train)
        # accuracy = accuracy_score(y_test, model.predict(X_test))

        print(f'Out-of-bag score estimate: {model.oob_score_:.3}', file=fout)
        # print(f'Mean accuracy score: {accuracy:.3}', file=fout)
        print(classification_report(y_train, model.predict(X_train)), file=fout)
        # print(classification_report(y_test, model.predict(X_test)), file=fout)

        fout.close()
