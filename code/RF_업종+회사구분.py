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
code_PATH = 'C:\\Users\\ialab\\Desktop\\T_Friend_data\\업종코드_매칭\\'
d18_train = 'train_test18.xlsx'
d17_train = 'train_test17.xlsx'
d17_18_train = 'train_test17_18.xlsx'
year17_data = '2017_data.xlsx'
new_data = 'total_17_18.xlsx'

train = pd.read_excel(code_PATH + new_data, encoding='utf-8')

X = train.loc[:, [ '사업자등록번호' ,'c', '업종코드']].copy()
X_label = train.loc[:, ['계정과목']].copy()
X_label.dropna(inplace=True)#빈곳 제거후 땡겨채운(inplace)
index = X_label['계정과목'].str.split(' ', n=1, expand=True) #expand - 별도의 열로 확장, n - 몇번째 ''
lists = list(index)
Y = index[0].copy()

#X : dataset Y : label

X['c'] = LabelEncoder().fit_transform(X['c'])#종류별로 column 나// LabelEncoder() - 0~7로 라벨을 바꿔줌(one-hot 전 단계), fit_transform() - 변경시킬 데이터를 받아주는 역할
X2 = pd.DataFrame(LabelBinarizer().fit_transform(X['c']), columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7']
                  , index=X.index) #X.index - X(데이터 정보가 들어있는)의 인덱스를 기준으로 매칭시킬것
X = pd.concat([X, X2], axis=1)#결합  axis=1 : column
X['회사구분'] = LabelEncoder().fit_transform(X['회사구분'])#종류별로 column 나// LabelEncoder() - 0~7로 라벨을 바꿔줌(one-hot 전 단계), fit_transform() - 변경시킬 데이터를 받아주는 역할
X3 = pd.DataFrame(LabelBinarizer().fit_transform(X['회사구분']), columns=['회사구분0', '회사구분1', '회사구분2', '회사구분3']
                  , index=X.index) #X.index - X(데이터 정보가 들어있는)의 인덱스를 기준으로 매칭시킬것
X = pd.concat([X, X3], axis=1)#결합  axis=1 : column


num = X['사업자등록번호'].str.split('-', n=2, expand=True)# n = 1 : 1번 분해
num['거래처번호'] = num[0].str.cat(num[1])
num['거래처번호'] = num['거래처번호'].str.cat(num[2])

X = pd.concat([X, num['거래처번호']], axis=1)


del(X['c'])
del(X['회사구분'])
del(X['사업자등록번호'])#사전작업이 끝난 원본 data 지움

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

model = RandomForestClassifier(n_estimators=35, oob_score=True, random_state=123456)#n_estimators=35<-라벨수, oob_score=True<-score출력, random_state=123456<-randum값 고정 시드
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

filename = '2018_RF_model.sav'
fout = open(code_PATH + 'acc4.txt', 'a')
pickle.dump(model, open(filename, 'wb'))
#model = DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_leaf=5).fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))

print(f'Out-of-bag score estimate: {model.oob_score_:.3}', file=fout)
print(f'Mean accuracy score: {accuracy:.3}', file=fout)
print(classification_report(y_train, model.predict(X_train)), file=fout)
print(classification_report(y_test, model.predict(X_test)), file=fout)

fout.close()

