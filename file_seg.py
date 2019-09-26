import pandas as pd

PATH = 'C:\\Users\\ialab\\Desktop\\T-Friend\\'
filename = 'test_file.xlsx'

df = pd.read_excel(PATH + filename, encoding='utf-8', index_col=0)
print(df.head())

#pre_data = df.loc[:, [name,name2]].astype('str')

## 매입/매춰 나누기
C_in = []
C_out = []

for i in range(len(df)):
    #print(df.loc[i,['CD_SCRP']].values)
    name = df.loc[i,['CD_SCRP']].item()
    name_list = list(name)
    last_num = len(name_list)-1
    if name_list[last_num] == 'n': C_in.append(i)
    else : C_out.append(i)

df_in = df.iloc[C_in,:]
df_in = df_in.reset_index()
df_in = df_in.drop(['index'], axis=1)

df_out = df.iloc[C_out,:]
df_out = df_out.reset_index()
df_out = df_out.drop(['index'], axis=1)

df_in.to_excel('in_file.xlsx', 'w', encoding='utf-8')
df_out.to_excel('out_file.xlsx', 'w', encoding='utf-8')

