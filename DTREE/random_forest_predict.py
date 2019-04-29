import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

PATH = 'c:\\3-f\\3friend_raw_data\\'
excel_PATH = 'c:\\3-f\\excel_data\\'
code_PATH = 'c:\\3-f\\Code\\'
d18_train = 'train_test.xlsx' # 2018 data
d17_train = 'train_test17.xlsx' # 2017 data
year17_data = '2017_data.xlsx'

train = pd.read_excel(code_PATH + d17_train, encoding='utf-8')

X = train.loc[:, ['회사코드', '사업자등록번호', 'c']].copy()
X_label = train.loc[:, ['계정과목']].copy()
X_label.dropna(inplace=True)
index = X_label['계정과목'].str.split(' ', n=1, expand=True)
lists = list(index)
Y = index[0].copy()

X['c'] = LabelEncoder().fit_transform(X['c'])
X2 = pd.DataFrame(LabelBinarizer().fit_transform(X['c']), columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7']
                  , index=X.index)
X = pd.concat([X, X2], axis=1)

num = X['사업자등록번호'].str.split('-', n=2, expand=True)
num['거래처번호'] = num[0].str.cat(num[1])
num['거래처번호'] = num['거래처번호'].str.cat(num[2])

X = pd.concat([X, num['거래처번호']], axis=1)

del(X['c'])
del(X['사업자등록번호'])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)


filename17 = '2017_RF_model.sav'
filename18 = '2018_RF_model.sav'
loaded_model = pickle.load(open(filename17, 'rb'))

predict = loaded_model.predict(X)
df = pd.DataFrame(predict)
df.to_excel('test_Test_test_test.xlsx')

result = loaded_model.score(X, Y)
fout = open('비교표.txt', 'a')
print('tr17, te 17 : %f' %result, file=fout)
fout.close()