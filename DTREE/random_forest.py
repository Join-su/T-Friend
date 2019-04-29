import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

raw_DATA = '3friend_raw_data.xlsx'
d18_train = 'train_test18.xlsx'
d17_train = 'train_test17.xlsx'
excel_PATH = "C:\\Git\\T-Friend\\3friend_raw_data\\"
img_PATH = "C:\\Git\\T-Friend\\img_data_predict"
predict_PATH = "C:\\Git\\T-Friend\\predict_excel\\"

train = pd.read_excel(excel_PATH + d17_train, encoding='utf-8')

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

model = RandomForestClassifier(n_estimators=35, oob_score=True, random_state=123456)
model.fit(X_train, y_train)


filename = '2017_RF_model.sav'
fout = open('2017_acc.txt', 'a')
pickle.dump(model, open(filename, 'wb'))
#model = DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_leaf=5).fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))

print(f'Out-of-bag score estimate: {model.oob_score_:.3}', file=fout)
print(f'Mean accuracy score: {accuracy:.3}', file=fout)
print(classification_report(y_train, model.predict(X_train)), file=fout)
print(classification_report(y_test, model.predict(X_test)), file=fout)

fout.close()