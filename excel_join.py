import pandas as pd

path = "/home/cent/Documents/github/T-friend/train_data/"
save_path = "/home/cent/Documents/github/T-friend/process/"

T = '계산서'

if T == '계산서' :

    filename_1 = 'e_train_data_test_1.xlsx'
    filename_2 = 'e_train_data_test_2.xlsx'

    df_1 = pd.read_excel(path + filename_1, encoding='utf-8', index_col=0)
    df_2 = pd.read_excel(path + filename_2, encoding='utf-8', index_col=0)

    df_2 = df_2.loc[:,["NO_BIZ_C", "NM_ITEM", "CD_ACCOUNT","NO_BIZ", "CD_INDUSTRY"]]
    df_1 = df_1.loc[:,["NO_BIZ_C", "NM_ITEM", "CD_ACCOUNT","NO_BIZ", "CD_INDUSTRY"]]

elif T == '영수증' :

    filename_1 = 'card_train_data_test_1.xlsx'
    filename_2 = 'card_train_data_test_2.xlsx'

    df_1 = pd.read_excel(path + filename_1, encoding='utf-8', index_col=0)
    df_2 = pd.read_excel(path + filename_2, encoding='utf-8', index_col=0)

    df_2 = df_2.loc[:, ["NO_BIZ_C", "TP_BIZ_C", "CD_ACCOUNT", "NO_BIZ"]]
    df_1 = df_1.loc[:, ["NO_BIZ_C", "TP_BIZ_C", "CD_ACCOUNT", "NO_BIZ"]]
buyer = "NO_BIZ"
seller = "NO_BIZ_C"


try:
    buyer_raw_data = df_1[seller].str.split('-', n=2, expand=True)
    buyer_raw_data[seller] = buyer_raw_data[0].str.cat(buyer_raw_data[1])
    buyer_raw_data[seller] = buyer_raw_data[seller].str.cat(buyer_raw_data[2]).copy()
    del (buyer_raw_data[0])
    del (buyer_raw_data[1])
    del (buyer_raw_data[2])
    buyer_raw_data = buyer_raw_data.astype('str')

    del (df_1[seller])
    df_1[seller] = buyer_raw_data[seller].astype('str')
except AttributeError:
    print('-부호 없음')


try:
    buyer_raw_data = df_1[buyer].str.split('-', n=2, expand=True)
    buyer_raw_data[buyer] = buyer_raw_data[0].str.cat(buyer_raw_data[1])
    buyer_raw_data[buyer] = buyer_raw_data[buyer].str.cat(buyer_raw_data[2]).copy()
    del (buyer_raw_data[0])
    del (buyer_raw_data[1])
    del (buyer_raw_data[2])
    buyer_raw_data = buyer_raw_data.astype('str')

    del (df_1[buyer])
    df_1[buyer] = buyer_raw_data[buyer].astype('str')
except AttributeError:
    print('-부호 없음')

try:
    buyer_raw_data = df_2[seller].str.split('-', n=2, expand=True)
    buyer_raw_data[seller] = buyer_raw_data[0].str.cat(buyer_raw_data[1])
    buyer_raw_data[seller] = buyer_raw_data[seller].str.cat(buyer_raw_data[2]).copy()
    del (buyer_raw_data[0])
    del (buyer_raw_data[1])
    del (buyer_raw_data[2])
    buyer_raw_data = buyer_raw_data.astype('str')

    del (df_2[seller])
    df_2[seller] = buyer_raw_data[seller].astype('str')
except AttributeError:
    print('-부호 없음')


try:
    buyer_raw_data = df_2[buyer].str.split('-', n=2, expand=True)
    buyer_raw_data[buyer] = buyer_raw_data[0].str.cat(buyer_raw_data[1])
    buyer_raw_data[buyer] = buyer_raw_data[buyer].str.cat(buyer_raw_data[2]).copy()
    del (buyer_raw_data[0])
    del (buyer_raw_data[1])
    del (buyer_raw_data[2])
    buyer_raw_data = buyer_raw_data.astype('str')

    del (df_2[buyer])
    df_2[buyer] = buyer_raw_data[buyer].astype('str')
except AttributeError:
    print('-부호 없음')

df_2[buyer] = df_2[buyer].astype('str')
df_1[buyer] = df_1[buyer].astype('str')

target = "CD_ACCOUNT"

try:
    buyer_raw_data = df_1[target].str.split(' ', n=1, expand=True)
    buyer_raw_data[target] = buyer_raw_data[0]
    del (buyer_raw_data[0])
    del (buyer_raw_data[1])
    buyer_raw_data = buyer_raw_data.astype('str')

    del (df_1[target])
    df_1[target] = buyer_raw_data[target].astype('str')
except AttributeError:
    print('pass')

try:
    buyer_raw_data = df_2[target].str.split(' ', n=1, expand=True)
    buyer_raw_data[target] = buyer_raw_data[0]
    del (buyer_raw_data[0])
    del (buyer_raw_data[1])
    buyer_raw_data = buyer_raw_data.astype('str')

    del (df_2[target])
    df_2[target] = buyer_raw_data[target].astype('str')
except AttributeError:
    print('pass')

df_2[target] = df_2[target].astype('str')
df_1[target] = df_1[target].astype('str')

print("df_1.head() : ",df_1.head())
print("df_2.head() : ",df_2.head())

#df_result = pd.merge(df_1,df_2, on='CD_ACCOUNT',how="outer", left_index=True, right_index=True)
df_result = pd.concat([df_1,df_2])
df_result = df_result.reset_index()
df_result = df_result.drop(['index'], axis=1)


if T == '계산서' :
    df_result.to_excel(save_path + '12_file.xlsx', 'w', encoding='utf-8')
elif T == '영수증' :
    df_result.to_excel(save_path + '34_file.xlsx', 'w', encoding='utf-8')
